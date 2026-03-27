"""
Generate an audio continuation by first synthesizing a prompt clip from text
with MeloTTS, then feeding that prompt through the EnCodec-token GPT pipeline.
"""

import os
import pickle
import sys
from contextlib import nullcontext

import torch
import torchaudio

from audio_codec import decode_codes
from audio_codec import encode_audio_file
from audio_codec import generate_audio_tokens
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_codec import unflatten_codes
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = "out-audio-librispeech"
prompt_text = ""
melotts_repo = "/tmp/MeloTTS"
tts_language = "EN"
tts_speaker = "EN-US"
tts_speed = 1.0
tts_device = "cpu"
prompt_audio = ""
prompt_codec_audio = ""
output_wav = ""
continuation_wav = ""
prompt_max_seconds = 0.0
max_new_seconds = 1.0
temperature = 0.9
top_k = 100
seed = 1337
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def import_melotts_tts():
    try:
        from melo.api import TTS
        return TTS
    except ImportError:
        pass

    if melotts_repo and os.path.isdir(melotts_repo):
        if melotts_repo not in sys.path:
            sys.path.insert(0, melotts_repo)
        from melo.api import TTS
        return TTS

    raise ImportError(
        "Could not import MeloTTS. Either install `melo` in the active environment "
        "or set --melotts_repo to a local clone of https://github.com/myshell-ai/MeloTTS ."
    )


def load_checkpoint_model():
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model, checkpoint


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def maybe_trim_audio(path: str, max_seconds: float) -> str:
    if max_seconds <= 0:
        return path

    wav, sample_rate = torchaudio.load(path)
    max_samples = int(max_seconds * sample_rate)
    if wav.size(-1) <= max_samples:
        return path

    trimmed = wav[..., :max_samples]
    root, ext = os.path.splitext(path)
    trimmed_path = f"{root}_trimmed{ext}"
    ensure_parent_dir(trimmed_path)
    torchaudio.save(trimmed_path, trimmed, sample_rate)
    return trimmed_path


def default_output_paths():
    artifact_dir = os.path.join(out_dir, "tts_prompt_tests")
    os.makedirs(artifact_dir, exist_ok=True)
    base = "generated_from_text"

    paths = {
        "prompt_audio": prompt_audio or os.path.join(artifact_dir, f"{base}_prompt.wav"),
        "prompt_codec_audio": prompt_codec_audio or os.path.join(artifact_dir, f"{base}_prompt_codec.wav"),
        "output_wav": output_wav or os.path.join(artifact_dir, f"{base}_full.wav"),
        "continuation_wav": continuation_wav or os.path.join(artifact_dir, f"{base}_continuation.wav"),
    }
    return paths


def synthesize_prompt(text: str, output_path: str):
    if not text.strip():
        raise ValueError("Set --prompt_text='some text to speak'")

    TTS = import_melotts_tts()
    model = TTS(language=tts_language, device=tts_device)
    speaker_ids = model.hps.data.spk2id
    if tts_speaker not in speaker_ids:
        available = ", ".join(sorted(speaker_ids))
        raise ValueError(f"Unknown tts_speaker={tts_speaker!r}. Available speakers: {available}")

    ensure_parent_dir(output_path)
    model.tts_to_file(
        text,
        speaker_ids[tts_speaker],
        output_path,
        speed=tts_speed,
        quiet=True,
    )
    return output_path


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

paths = default_output_paths()
prompt_path = synthesize_prompt(prompt_text, paths["prompt_audio"])
prompt_path = maybe_trim_audio(prompt_path, prompt_max_seconds)

model, checkpoint = load_checkpoint_model()
dataset = checkpoint.get("config", {}).get("dataset", "audio_codec")
meta_path = os.path.join("data", dataset, "meta.pkl")
with open(meta_path, "rb") as handle:
    meta = pickle.load(handle)

codec = load_encodec_model(
    model_name=meta["model_name"],
    bandwidth=meta["bandwidth"],
    device="cpu",
)
prompt_batch = encode_audio_file(
    prompt_path,
    codec,
    device="cpu",
    codebook_size=meta["codebook_size"],
)

tokens_per_second = meta["tokens_per_second"]
max_new_tokens = int(max_new_seconds * tokens_per_second)
num_codebooks = meta["num_codebooks"]
codebook_size = meta["codebook_size"]

x = torch.tensor(prompt_batch.flattened_tokens.tolist(), dtype=torch.long, device=device)[None, ...]
if x.size(1) > model.config.block_size:
    prompt_seconds = x.size(1) / tokens_per_second
    visible_seconds = model.config.block_size / tokens_per_second
    print(
        f"warning: prompt is {prompt_seconds:.2f}s but block_size only covers "
        f"{visible_seconds:.2f}s; generation is conditioned on the tail only"
    )

with torch.no_grad():
    with ctx:
        y = generate_audio_tokens(
            model,
            x,
            max_new_tokens=max_new_tokens,
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
prompt_wav_codec = decode_codes(codec, prompt_batch.codes, device="cpu")
full_wav = decode_codes(codec, codes, device="cpu")
prompt_num_samples = prompt_wav_codec.size(-1)
continuation_only = full_wav[:, prompt_num_samples:]

save_waveform(paths["prompt_codec_audio"], prompt_wav_codec, meta["sample_rate"])
save_waveform(paths["output_wav"], full_wav, meta["sample_rate"])
save_waveform(paths["continuation_wav"], continuation_only, meta["sample_rate"])

print(f"TTS prompt saved to {prompt_path}")
print(f"Codec prompt saved to {paths['prompt_codec_audio']}")
print(f"Full generated audio saved to {paths['output_wav']}")
print(f"Continuation-only audio saved to {paths['continuation_wav']}")
