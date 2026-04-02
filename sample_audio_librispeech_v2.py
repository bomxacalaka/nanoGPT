"""
Test the LibriSpeech v2 audio continuation model with a prompt chosen from
an indexed dataset clip, a random dataset clip, or an explicit audio file.
"""

import json
import os
import pickle
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

from audio_codec import decode_codes
from audio_codec import encode_waveform
from audio_codec import generate_audio_tokens
from audio_codec import load_audio
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_codec import unflatten_codes
from model import GPTConfig, GPT


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

# -----------------------------------------------------------------------------
out_dir = "out-audio-librispeech-v2"
model_preset = ""
prompt_audio = ""
prompt_dataset_split = "val"
prompt_index = -1
random_prompt = False
prompt_max_seconds = 2.0
max_new_seconds = 3.0
max_new_tokens = -1
temperature = 0.8
top_k = 50
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
output_subdir = "prompt_tests"
output_prefix = ""
list_offset = 0
list_count = 0
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def load_checkpoint_model():
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model, checkpoint


def dir_to_model_preset(path: Path) -> str:
    name = path.name
    if name.startswith("out-"):
        name = name[4:]
    return name.replace("-", "_")


def discover_audio_model_checkpoints() -> list[dict]:
    items = []
    root = Path(".")
    for ckpt_path in sorted(root.glob("out-*/ckpt.pt")):
        out_dir_path = ckpt_path.parent
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue

        dataset = checkpoint.get("config", {}).get("dataset", "")
        meta_path = root / "data" / dataset / "meta.pkl"
        if not meta_path.is_file():
            continue

        try:
            with meta_path.open("rb") as handle:
                meta = pickle.load(handle)
        except Exception:
            continue

        if meta.get("dataset_type") != "audio_codec":
            continue

        items.append(
            {
                "preset": dir_to_model_preset(out_dir_path),
                "out_dir": str(out_dir_path),
                "dataset": dataset,
            }
        )
    return items


def resolve_out_dir() -> str:
    if not model_preset:
        return out_dir
    discovered = {item["preset"]: item for item in discover_audio_model_checkpoints()}
    if model_preset not in discovered:
        available = ", ".join(sorted(discovered))
        raise ValueError(f"Unknown model_preset={model_preset!r}. Available presets: {available}")
    return discovered[model_preset]["out_dir"]


def iter_audio_files(root_dir: str) -> list[str]:
    root = Path(root_dir)
    files = [
        str(path)
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    return files


def get_split_dir(meta: dict) -> str:
    split = prompt_dataset_split.lower()
    if split == "val":
        root_dir = meta.get("val_input_dir") or meta.get("input_dir")
    elif split == "train":
        root_dir = meta.get("input_dir")
    else:
        raise ValueError("Set --prompt_dataset_split to 'train' or 'val'")

    if not root_dir:
        raise ValueError(f"No dataset directory recorded for split={split!r}")
    return root_dir


def read_librispeech_transcript(audio_path: str) -> str:
    audio_file = Path(audio_path)
    utterance_id = audio_file.stem
    parts = utterance_id.split("-")
    if len(parts) < 3:
        return ""

    transcript_path = audio_file.parent / f"{parts[0]}-{parts[1]}.trans.txt"
    if not transcript_path.is_file():
        return ""

    prefix = f"{utterance_id} "
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                return line[len(prefix):].strip()
    return ""


def load_manifest_prompt_text(dataset: str, meta: dict) -> dict[str, str]:
    candidates = [
        Path("data") / dataset / "raw" / "manifest.jsonl",
        Path(meta.get("input_dir", "")).resolve().parent / "manifest.jsonl" if meta.get("input_dir") else None,
        Path(meta.get("val_input_dir", "")).resolve().parent / "manifest.jsonl" if meta.get("val_input_dir") else None,
    ]
    mapping = {}
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                path = record.get("path")
                text = record.get("text")
                if path and text:
                    mapping[os.path.abspath(path)] = text
        if mapping:
            return mapping
    return mapping


def get_prompt_text(audio_path: str, prompt_text_by_path: dict[str, str]) -> str:
    abs_path = os.path.abspath(audio_path)
    if abs_path in prompt_text_by_path:
        return prompt_text_by_path[abs_path]
    return read_librispeech_transcript(audio_path)


def get_dataset_files(meta: dict) -> tuple[str, list[str]]:
    root_dir = get_split_dir(meta)
    files = iter_audio_files(root_dir)
    if not files:
        raise ValueError(f"No audio files found under {root_dir}")
    return root_dir, files


def clamp_window_offset(offset: int, total_count: int) -> int:
    if total_count <= 0:
        return 0
    if list_count <= 0:
        return max(0, min(offset, total_count - 1))

    max_offset = max(0, total_count - list_count)
    return max(0, min(offset, max_offset))


def print_dataset_window(files: list[str], root_dir: str, offset: int, prompt_text_by_path: dict[str, str]):
    if list_count <= 0:
        return

    start = clamp_window_offset(offset, len(files))
    end = min(len(files), start + list_count)
    print("")
    print(f"Dataset prompts {start}..{max(start, end) - 1} of {len(files) - 1} from {root_dir}")
    for idx in range(start, end):
        rel_path = os.path.relpath(files[idx], root_dir)
        prompt_text = get_prompt_text(files[idx], prompt_text_by_path)
        if prompt_text:
            print(f"[{idx}] {rel_path} :: {prompt_text}")
        else:
            print(f"[{idx}] {rel_path}")
    if end < len(files):
        print(f"... {len(files) - end} more files not shown")
    print("Commands: [index] generate | n next page | p previous page | r random | q quit")


def choose_prompt_audio(files: list[str]) -> tuple[str, int, str, int]:
    if prompt_audio:
        return prompt_audio, -1, "custom", 0

    if prompt_index >= 0:
        if prompt_index >= len(files):
            raise ValueError(
                f"prompt_index={prompt_index} is out of range for split={prompt_dataset_split!r} "
                f"with {len(files)} files"
            )
        return files[prompt_index], prompt_index, prompt_dataset_split.lower(), len(files)

    if random_prompt:
        rng = random.Random(seed)
        chosen_index = rng.randrange(len(files))
        return files[chosen_index], chosen_index, prompt_dataset_split.lower(), len(files)

    if not sys.stdin.isatty():
        raise ValueError(
            "No prompt selected. Set --prompt_audio=/path/to/file, "
            "--prompt_index=N, or --random_prompt=True."
        )

    print(f"Found {len(files)} files in split={prompt_dataset_split!r}.")
    user_value = input(f"Enter prompt index [0-{len(files) - 1}] or press Enter for random: ").strip()
    if not user_value:
        rng = random.Random(seed)
        chosen_index = rng.randrange(len(files))
        return files[chosen_index], chosen_index, prompt_dataset_split.lower(), len(files)

    chosen_index = int(user_value)
    if chosen_index < 0 or chosen_index >= len(files):
        raise ValueError(f"Prompt index must be between 0 and {len(files) - 1}")
    return files[chosen_index], chosen_index, prompt_dataset_split.lower(), len(files)


def trim_waveform(wav: torch.Tensor, sample_rate: int, max_seconds: float) -> torch.Tensor:
    if max_seconds <= 0:
        return wav

    max_samples = int(max_seconds * sample_rate)
    if wav.size(-1) <= max_samples:
        return wav
    return wav[..., :max_samples]


def default_output_paths(selected_prompt_path: str, selected_index: int, selected_split: str):
    artifact_dir = os.path.join(out_dir, output_subdir)
    os.makedirs(artifact_dir, exist_ok=True)

    if output_prefix:
        if selected_index >= 0:
            stem = Path(selected_prompt_path).stem
            base = f"{output_prefix}_{selected_split}_{selected_index:04d}_{stem}"
        else:
            base = output_prefix
    elif selected_index >= 0:
        stem = Path(selected_prompt_path).stem
        base = f"{selected_split}_{selected_index:04d}_{stem}"
    else:
        base = Path(selected_prompt_path).stem

    return {
        "prompt_wav": os.path.join(artifact_dir, f"{base}_prompt.wav"),
        "output_wav": os.path.join(artifact_dir, f"{base}_full.wav"),
        "continuation_wav": os.path.join(artifact_dir, f"{base}_continuation.wav"),
        "metadata_json": os.path.join(artifact_dir, f"{base}_metadata.json"),
    }


def resolve_generation_length(tokens_per_second: int, num_codebooks: int) -> tuple[int, int, float]:
    if max_new_tokens >= 0:
        requested_tokens = max_new_tokens
    else:
        requested_tokens = int(max_new_seconds * tokens_per_second)

    effective_tokens = requested_tokens - (requested_tokens % num_codebooks)
    effective_seconds = effective_tokens / tokens_per_second
    return requested_tokens, effective_tokens, effective_seconds


def run_generation(
    model,
    codec,
    meta: dict,
    dataset: str,
    prompt_text_by_path: dict[str, str],
    selected_prompt_path: str,
    selected_index: int,
    selected_split: str,
    split_file_count: int,
):
    paths = default_output_paths(selected_prompt_path, selected_index, selected_split)
    prompt_text = get_prompt_text(selected_prompt_path, prompt_text_by_path)

    prompt_wav_raw, prompt_sample_rate = load_audio(selected_prompt_path)
    prompt_wav_raw = trim_waveform(prompt_wav_raw, prompt_sample_rate, prompt_max_seconds)
    prompt_batch = encode_waveform(
        codec,
        prompt_wav_raw,
        prompt_sample_rate,
        device="cpu",
        codebook_size=meta["codebook_size"],
    )

    tokens_per_second = meta["tokens_per_second"]
    num_codebooks = meta["num_codebooks"]
    codebook_size = meta["codebook_size"]
    requested_new_tokens, effective_new_tokens, effective_new_seconds = resolve_generation_length(
        tokens_per_second=tokens_per_second,
        num_codebooks=num_codebooks,
    )

    x = torch.tensor(prompt_batch.flattened_tokens.tolist(), dtype=torch.long, device=device)[None, ...]
    prompt_token_count = x.size(1)
    context_seconds = model.config.block_size / tokens_per_second
    prompt_seconds = prompt_token_count / tokens_per_second
    print("")
    print(f"Selected prompt: {selected_prompt_path}")
    if selected_index >= 0:
        print(f"Prompt index: {selected_index} / {split_file_count - 1} ({selected_split})")
    if prompt_text:
        print(f"Prompt text: {prompt_text}")
    print(
        f"Prompt length: {prompt_token_count} tokens ({prompt_seconds:.2f}s). "
        f"Model context window: {model.config.block_size} tokens ({context_seconds:.2f}s)."
    )
    if x.size(1) > model.config.block_size:
        print(
            f"warning: prompt is {prompt_seconds:.2f}s but block_size only covers "
            f"{context_seconds:.2f}s; generation is conditioned on the tail only"
        )

    if max_new_tokens >= 0:
        print(
            f"Requested continuation: {requested_new_tokens} tokens. "
            f"Effective continuation: {effective_new_tokens} tokens ({effective_new_seconds:.2f}s)."
        )
    else:
        print(
            f"Requested continuation: {max_new_seconds:.2f}s. "
            f"Effective continuation: {effective_new_tokens} tokens ({effective_new_seconds:.2f}s)."
        )

    if requested_new_tokens != effective_new_tokens:
        print(
            f"note: rounded continuation down by {requested_new_tokens - effective_new_tokens} tokens "
            f"to align with {num_codebooks} interleaved codebooks"
        )

    with torch.no_grad():
        with ctx:
            y = generate_audio_tokens(
                model,
                x,
                max_new_tokens=effective_new_tokens,
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
                temperature=temperature,
                top_k=top_k,
            )

    all_tokens = y[0].tolist()
    valid_token_count = len(all_tokens) - (len(all_tokens) % num_codebooks)
    all_tokens = all_tokens[:valid_token_count]

    codes = unflatten_codes(
        torch.tensor(all_tokens, dtype=torch.long),
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )
    prompt_wav = decode_codes(codec, prompt_batch.codes, device="cpu")
    full_wav = decode_codes(codec, codes, device="cpu")
    prompt_num_samples = prompt_wav.size(-1)
    continuation_only = full_wav[:, prompt_num_samples:]

    save_waveform(paths["prompt_wav"], prompt_wav, meta["sample_rate"])
    save_waveform(paths["output_wav"], full_wav, meta["sample_rate"])

    metadata = {
        "checkpoint_dir": out_dir,
        "dataset": dataset,
        "prompt_source_path": os.path.abspath(selected_prompt_path),
        "prompt_split": selected_split,
        "prompt_index": selected_index,
        "prompt_split_file_count": split_file_count,
        "prompt_text": prompt_text,
        "prompt_max_seconds": prompt_max_seconds,
        "max_new_seconds": max_new_seconds,
        "requested_new_tokens": requested_new_tokens,
        "effective_new_tokens": effective_new_tokens,
        "effective_new_seconds": effective_new_seconds,
        "temperature": temperature,
        "top_k": top_k,
        "sample_rate": meta["sample_rate"],
        "tokens_per_second": tokens_per_second,
        "num_codebooks": num_codebooks,
        "model_block_size": model.config.block_size,
        "model_context_seconds": context_seconds,
        "prompt_token_count": int(prompt_token_count),
        "generated_token_count": int(len(all_tokens)),
        "prompt_audio_path": os.path.abspath(paths["prompt_wav"]),
        "full_audio_path": os.path.abspath(paths["output_wav"]),
    }

    if continuation_only.size(-1) > 0:
        save_waveform(paths["continuation_wav"], continuation_only, meta["sample_rate"])
        metadata["continuation_audio_path"] = os.path.abspath(paths["continuation_wav"])
    else:
        metadata["continuation_audio_path"] = ""

    with open(paths["metadata_json"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Prompt audio saved to {paths['prompt_wav']}")
    print(f"Full generated audio saved to {paths['output_wav']}")
    if continuation_only.size(-1) > 0:
        print(f"Continuation-only audio saved to {paths['continuation_wav']}")
    else:
        print("Continuation-only audio not saved because max_new_seconds produced no new samples")
    print(f"Run metadata saved to {paths['metadata_json']}")


def run_interactive_prompt_loop(
    model,
    codec,
    meta: dict,
    dataset: str,
    root_dir: str,
    files: list[str],
    prompt_text_by_path: dict[str, str],
):
    offset = clamp_window_offset(list_offset, len(files))
    rng = random.Random(seed)
    selected_split = prompt_dataset_split.lower()

    while True:
        print_dataset_window(files, root_dir, offset, prompt_text_by_path)
        try:
            user_value = input("Select prompt index or command: ").strip().lower()
        except EOFError:
            print("")
            print("EOF received, exiting interactive prompt loop.")
            return

        if user_value in {"q", "quit", "exit"}:
            print("Exiting interactive prompt loop.")
            return

        if user_value in {"n", "next"}:
            offset = clamp_window_offset(offset + max(1, list_count), len(files))
            continue

        if user_value in {"p", "prev"}:
            offset = clamp_window_offset(offset - max(1, list_count), len(files))
            continue

        if user_value in {"r", "rand", "random"}:
            chosen_index = rng.randrange(len(files))
            if list_count > 0:
                offset = clamp_window_offset((chosen_index // list_count) * list_count, len(files))
            run_generation(
                model=model,
                codec=codec,
                meta=meta,
                dataset=dataset,
                prompt_text_by_path=prompt_text_by_path,
                selected_prompt_path=files[chosen_index],
                selected_index=chosen_index,
                selected_split=selected_split,
                split_file_count=len(files),
            )
            continue

        if not user_value:
            continue

        try:
            chosen_index = int(user_value)
        except ValueError:
            print("Invalid command. Enter an index, `n`, `p`, `r`, or `q`.")
            continue

        if chosen_index < 0 or chosen_index >= len(files):
            print(f"Prompt index must be between 0 and {len(files) - 1}")
            continue

        run_generation(
            model=model,
            codec=codec,
            meta=meta,
            dataset=dataset,
            prompt_text_by_path=prompt_text_by_path,
            selected_prompt_path=files[chosen_index],
            selected_index=chosen_index,
            selected_split=selected_split,
            split_file_count=len(files),
        )


def main():
    global out_dir
    global ctx

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    out_dir = resolve_out_dir()
    model, checkpoint = load_checkpoint_model()
    dataset = checkpoint.get("config", {}).get("dataset", "audio_codec")
    meta_path = os.path.join("data", dataset, "meta.pkl")
    with open(meta_path, "rb") as handle:
        meta = pickle.load(handle)
    prompt_text_by_path = load_manifest_prompt_text(dataset, meta)

    codec = load_encodec_model(
        model_name=meta["model_name"],
        bandwidth=meta["bandwidth"],
        device="cpu",
    )

    try:
        if prompt_audio:
            run_generation(
                model=model,
                codec=codec,
                meta=meta,
                dataset=dataset,
                prompt_text_by_path=prompt_text_by_path,
                selected_prompt_path=prompt_audio,
                selected_index=-1,
                selected_split="custom",
                split_file_count=0,
            )
        else:
            root_dir, files = get_dataset_files(meta)
            if list_count > 0 and prompt_index < 0 and not random_prompt and sys.stdin.isatty():
                run_interactive_prompt_loop(
                    model=model,
                    codec=codec,
                    meta=meta,
                    dataset=dataset,
                    root_dir=root_dir,
                    files=files,
                    prompt_text_by_path=prompt_text_by_path,
                )
            else:
                selected_prompt_path, selected_index, selected_split, split_file_count = choose_prompt_audio(files)
                run_generation(
                    model=model,
                    codec=codec,
                    meta=meta,
                    dataset=dataset,
                    prompt_text_by_path=prompt_text_by_path,
                    selected_prompt_path=selected_prompt_path,
                    selected_index=selected_index,
                    selected_split=selected_split,
                    split_file_count=split_file_count,
                )
    except KeyboardInterrupt:
        print("")
        print("Stopped by user.")


if __name__ == "__main__":
    main()
