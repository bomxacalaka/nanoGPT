"""
Generate a synthetic spoken-math dataset with MeloTTS.

Default corpus:
- addition only
- configurable operand range
- configurable spoken templates
- all available English MeloTTS voices by default
- split by operand pair so validation uses unseen sums, not just unseen voices/templates
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def import_melotts_tts(melotts_repo: str):
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
        "Could not import MeloTTS. Install `melo` in the active environment or set "
        "--melotts_repo to a local clone of https://github.com/myshell-ai/MeloTTS ."
    )


def number_to_words(n: int) -> str:
    ones = [
        "zero", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n < 20:
        return ones[n]
    if n < 100:
        t = n // 10
        o = n % 10
        return tens[t] if o == 0 else f"{tens[t]}-{ones[o]}"
    raise ValueError(f"number_to_words only supports values < 100 here, got {n}")


ADDITION_TEMPLATE_GROUPS = {
    "basic": [
        "{a} plus {b} equals {result}",
        "{a} plus {b} is {result}",
    ],
    "spoken": [
        "what is {a} plus {b} it is {result}",
        "if you add {a} and {b} the answer is {result}",
        "adding {a} and {b} gives {result}",
        "the sum of {a} and {b} is {result}",
    ],
    "teacher": [
        "compute {a} plus {b} equals {result}",
        "calculate {a} plus {b} the result is {result}",
        "solve {a} plus {b} equals {result}",
    ],
}


def resolve_templates(template_names: list[str]) -> list[str]:
    templates = []
    for name in template_names:
        if name == "all":
            for group in ADDITION_TEMPLATE_GROUPS.values():
                templates.extend(group)
            continue
        if name not in ADDITION_TEMPLATE_GROUPS:
            available = ", ".join(["all", *sorted(ADDITION_TEMPLATE_GROUPS)])
            raise ValueError(f"Unknown template group {name!r}. Available: {available}")
        templates.extend(ADDITION_TEMPLATE_GROUPS[name])
    deduped = []
    seen = set()
    for template in templates:
        if template not in seen:
            seen.add(template)
            deduped.append(template)
    return deduped


def build_records(min_number: int, max_number: int, speakers: list[str], templates: list[str]):
    records = []
    pair_index = 0
    record_id = 0
    for a in range(min_number, max_number + 1):
        for b in range(min_number, max_number + 1):
            total = a + b
            split = "val" if pair_index % 10 == 0 else "train"
            words = {
                "a": number_to_words(a),
                "b": number_to_words(b),
                "result": number_to_words(total),
            }
            for template_id, template in enumerate(templates):
                text = template.format(**words)
                for speaker in speakers:
                    records.append(
                        {
                            "id": record_id,
                            "pair_id": pair_index,
                            "template_id": template_id,
                            "template": template,
                            "split": split,
                            "speaker": speaker,
                            "a": a,
                            "b": b,
                            "result": total,
                            "text": text,
                        }
                    )
                    record_id += 1
            pair_index += 1
    return records


def synthesize_records(model, speaker_id: int, records: list[dict], speed: float, batch_size: int):
    if batch_size <= 1:
        for record in records:
            audio = model.tts_to_file(
                record["text"],
                speaker_id,
                output_path=None,
                speed=speed,
                quiet=True,
            )
            yield record, audio
        return

    language = model.language
    device = model.device
    hop_length = model.hps.data.hop_length

    total = len(records)
    for start in range(0, total, batch_size):
        batch_records = records[start : start + batch_size]
        can_batch = True

        for record in batch_records:
            pieces = model.split_sentences_into_pieces(record["text"], language, quiet=True)
            if len(pieces) != 1:
                can_batch = False
                break
            text = pieces[0]
            if language in ["EN", "ZH_MIX_EN"]:
                import re
                text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        if not can_batch:
            for record in batch_records:
                audio = model.tts_to_file(
                    record["text"],
                    speaker_id,
                    output_path=None,
                    speed=speed,
                    quiet=True,
                )
                yield record, audio
            continue

        prepared = []
        for record in batch_records:
            pieces = model.split_sentences_into_pieces(record["text"], language, quiet=True)
            text = pieces[0]
            if language in ["EN", "ZH_MIX_EN"]:
                import re
                text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
            from melo import utils as melo_utils

            bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
                text,
                language,
                model.hps,
                device,
                model.symbol_to_id,
            )
            prepared.append(
                {
                    "record": record,
                    "phones": phones,
                    "tones": tones,
                    "lang_ids": lang_ids,
                    "bert": bert,
                    "ja_bert": ja_bert,
                }
            )

        max_len = max(item["phones"].size(0) for item in prepared)
        phone_batch = torch.zeros(len(prepared), max_len, dtype=torch.long, device=device)
        tone_batch = torch.zeros(len(prepared), max_len, dtype=torch.long, device=device)
        lang_batch = torch.zeros(len(prepared), max_len, dtype=torch.long, device=device)
        bert_batch = torch.zeros(len(prepared), 1024, max_len, dtype=torch.float32, device=device)
        ja_bert_batch = torch.zeros(len(prepared), 768, max_len, dtype=torch.float32, device=device)
        lengths = torch.zeros(len(prepared), dtype=torch.long, device=device)

        for idx, item in enumerate(prepared):
            length = item["phones"].size(0)
            lengths[idx] = length
            phone_batch[idx, :length] = item["phones"].to(device)
            tone_batch[idx, :length] = item["tones"].to(device)
            lang_batch[idx, :length] = item["lang_ids"].to(device)
            bert_batch[idx, :, :length] = item["bert"].to(device)
            ja_bert_batch[idx, :, :length] = item["ja_bert"].to(device)

        speakers = torch.full((len(prepared),), speaker_id, dtype=torch.long, device=device)
        with torch.no_grad():
            audio_batch, _, y_mask, _ = model.model.infer(
                phone_batch,
                lengths,
                speakers,
                tone_batch,
                lang_batch,
                bert_batch,
                ja_bert_batch,
                sdp_ratio=0.2,
                noise_scale=0.6,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
            )
        audio_batch = audio_batch.detach().cpu().float().numpy()
        frame_lengths = y_mask.squeeze(1).sum(dim=1).detach().cpu().numpy().astype(np.int64)
        sample_lengths = frame_lengths * hop_length

        for idx, item in enumerate(prepared):
            audio = audio_batch[idx, 0, : sample_lengths[idx]]
            audio = model.audio_numpy_concat([audio], sr=model.hps.data.sampling_rate, speed=speed)
            yield item["record"], audio

    if "cuda" in str(device):
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "raw"))
    parser.add_argument("--min_number", type=int, default=0)
    parser.add_argument("--max_number", type=int, default=14)
    parser.add_argument("--language", default="EN")
    parser.add_argument("--speakers", default="", help="Comma-separated speaker list. Empty means all speakers for the chosen language.")
    parser.add_argument(
        "--template_groups",
        default="basic,spoken",
        help="Comma-separated template groups. Available: all,basic,spoken,teacher",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of same-speaker utterances to synthesize per forward pass.")
    parser.add_argument("--melotts_repo", default="/tmp/MeloTTS")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    TTS = import_melotts_tts(args.melotts_repo)

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    model = TTS(language=args.language, device=args.device)
    speaker_ids = model.hps.data.spk2id
    if args.speakers:
        speakers = [speaker.strip() for speaker in args.speakers.split(",") if speaker.strip()]
    else:
        speakers = sorted(speaker_ids.keys())

    missing = [speaker for speaker in speakers if speaker not in speaker_ids]
    if missing:
        available = ", ".join(sorted(speaker_ids))
        raise SystemExit(f"Unknown speakers: {', '.join(missing)}. Available speakers: {available}")

    if args.min_number < 0 or args.max_number < args.min_number:
        raise SystemExit("Expected 0 <= min_number <= max_number")

    template_names = [item.strip() for item in args.template_groups.split(",") if item.strip()]
    templates = resolve_templates(template_names)
    records = build_records(args.min_number, args.max_number, speakers, templates)
    manifest_path = output_dir / "manifest.jsonl"
    expressions_path = output_dir / "expressions.txt"

    with manifest_path.open("w", encoding="utf-8") as manifest, expressions_path.open("w", encoding="utf-8") as exprs:
        records_by_speaker: dict[str, list[dict]] = {speaker: [] for speaker in speakers}
        for record in records:
            records_by_speaker[record["speaker"]].append(record)

        for speaker in speakers:
            speaker_records = records_by_speaker[speaker]
            speaker_slug = speaker.replace("/", "_").replace(" ", "_")
            pending = []
            for record in speaker_records:
                split_dir = train_dir if record["split"] == "train" else val_dir
                file_name = (
                    f"{record['id']:06d}_t{record['template_id']:02d}_{speaker_slug}_"
                    f"{record['a']:02d}_{record['b']:02d}_{record['result']:02d}.wav"
                )
                output_path = split_dir / file_name
                record["output_path"] = output_path

                if args.force or not output_path.exists():
                    pending.append(record)

            print(
                f"speaker {speaker}: {len(pending)} pending / {len(speaker_records)} total "
                f"(batch_size={args.batch_size}, device={args.device})",
                flush=True,
            )

            completed = 0
            pending_total = len(pending)
            for record, audio in synthesize_records(
                model,
                speaker_ids[speaker],
                pending,
                speed=args.speed,
                batch_size=args.batch_size,
            ):
                import soundfile
                soundfile.write(str(record["output_path"]), audio, model.hps.data.sampling_rate)
                completed += 1
                print(
                    f"gen {completed:4d}/{pending_total:4d} | {record['split']:5s} | "
                    f"{record['speaker']:10s} | {record['output_path'].name} | {record['text']}",
                    flush=True,
                )

            for record in speaker_records:
                output_path = record["output_path"]
                manifest_record = {
                    key: value
                    for key, value in record.items()
                    if key != "output_path"
                }
                manifest_record.update(
                    {
                        "language": args.language,
                        "speed": args.speed,
                        "batch_size": args.batch_size,
                        "path": str(output_path.resolve()),
                    }
                )
                manifest.write(json.dumps(manifest_record) + "\n")
                exprs.write(record["text"] + "\n")
            print(f"speaker {speaker}: manifest written", flush=True)

    train_count = sum(1 for record in records if record["split"] == "train")
    val_count = len(records) - train_count
    print(f"generated {len(records)} files total", flush=True)
    print(f"train files: {train_count}", flush=True)
    print(f"val files: {val_count}", flush=True)
    print(f"speakers: {', '.join(speakers)}", flush=True)
    print(f"template groups: {', '.join(template_names)}", flush=True)
    print(f"templates: {len(templates)}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
