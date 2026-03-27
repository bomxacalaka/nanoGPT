"""
Prepare a directory of audio files for next-token modeling over EnCodec tokens.

The intended pipeline is:
1. Load wav/flac/mp3 audio files.
2. Encode them into discrete EnCodec codebooks.
3. Flatten multi-codebook frames into one token stream.
4. Write train.bin, val.bin, and meta.pkl for nanoGPT-style training.
"""

import argparse
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from audio_codec import EncodecUnavailableError
from audio_codec import chunk_waveform
from audio_codec import encode_waveform
from audio_codec import load_audio
from audio_codec import load_encodec_model


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing source audio files.")
    parser.add_argument("--val_input_dir", default="", help="Optional separate validation audio directory.")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent), help="Directory for train.bin/val.bin/meta.pkl.")
    parser.add_argument("--model_name", default="encodec_24khz", choices=["encodec_24khz"])
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--chunk_seconds", type=float, default=1.0, help="Audio chunk size before codec encoding.")
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument("--max_files", type=int, default=0, help="Optional cap for debugging.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codebook_size", type=int, default=1024)
    return parser.parse_args()


def iter_audio_files(input_dir: str, max_files: int):
    files = [
        path
        for path in Path(input_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort()
    if max_files > 0:
        files = files[:max_files]
    return files


def split_files(files, val_frac: float, seed: int):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    if not files:
        return [], []

    val_count = max(1, math.ceil(len(files) * val_frac)) if len(files) > 1 else 0
    val_files = files[:val_count]
    train_files = files[val_count:]
    if not train_files:
        train_files, val_files = files, []
    return train_files, val_files


def encode_files(files, model, args):
    tokens = []
    total_chunks = 0
    separator_token_id = None

    chunk_num_samples = int(args.chunk_seconds * model.sample_rate) if args.chunk_seconds > 0 else None

    for path in files:
        wav, sample_rate = load_audio(str(path))
        for chunk in chunk_waveform(wav, chunk_num_samples):
            batch = encode_waveform(
                model,
                chunk,
                sample_rate,
                device=args.device,
                codebook_size=args.codebook_size,
            )
            if separator_token_id is None:
                separator_token_id = batch.codebook_size * batch.num_codebooks
            tokens.append(batch.flattened_tokens.numpy())
            tokens.append(np.array([separator_token_id], dtype=np.int64))
            total_chunks += 1

    if not tokens:
        return np.array([], dtype=np.uint16), total_chunks, separator_token_id

    flat = np.concatenate(tokens)
    if args.codebook_size * batch.num_codebooks <= np.iinfo(np.uint16).max:
        flat = flat.astype(np.uint16)
    else:
        flat = flat.astype(np.uint32)
    return flat, total_chunks, separator_token_id


def main():
    args = parse_args()

    try:
        model = load_encodec_model(
            model_name=args.model_name,
            bandwidth=args.bandwidth,
            device=args.device,
        )
    except EncodecUnavailableError as exc:
        raise SystemExit(str(exc))

    if args.val_input_dir:
        train_files = iter_audio_files(args.input_dir, args.max_files)
        val_files = iter_audio_files(args.val_input_dir, args.max_files)
        files = train_files + val_files
        if not train_files:
            raise SystemExit(f"No training audio files found under {args.input_dir}")
        if not val_files:
            raise SystemExit(f"No validation audio files found under {args.val_input_dir}")
    else:
        files = iter_audio_files(args.input_dir, args.max_files)
        if not files:
            raise SystemExit(f"No audio files found under {args.input_dir}")
        train_files, val_files = split_files(files, args.val_frac, args.seed)

    print(f"Found {len(files)} source files")
    print(f"Train files: {len(train_files)} | Val files: {len(val_files)}")

    train_tokens, train_chunks, separator_token_id = encode_files(train_files, model, args)
    val_tokens, val_chunks, _ = encode_files(val_files, model, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_tokens.tofile(output_dir / "train.bin")
    val_tokens.tofile(output_dir / "val.bin")

    num_codebooks = {1.5: 2, 3.0: 4, 6.0: 8, 12.0: 16, 24.0: 32}.get(float(args.bandwidth))

    meta = {
        "dataset_type": "audio_codec",
        "codec": "encodec",
        "model_name": args.model_name,
        "bandwidth": float(args.bandwidth),
        "sample_rate": model.sample_rate,
        "channels": model.channels,
        "frame_rate": model.frame_rate,
        "tokens_per_second": model.frame_rate * num_codebooks if num_codebooks is not None else None,
        "codebook_size": args.codebook_size,
        "num_codebooks": num_codebooks,
        "flattening": "time_major_with_codebook_offsets",
        "separator_token_id": separator_token_id,
        "vocab_size": (args.codebook_size * num_codebooks + 1) if num_codebooks is not None else None,
        "data_dtype": str(train_tokens.dtype),
        "input_dir": str(Path(args.input_dir).resolve()),
        "val_input_dir": str(Path(args.val_input_dir).resolve()) if args.val_input_dir else "",
        "num_source_files": len(files),
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_chunks": train_chunks,
        "val_chunks": val_chunks,
    }
    with open(output_dir / "meta.pkl", "wb") as handle:
        pickle.dump(meta, handle)

    print(f"train tokens: {len(train_tokens):,}")
    print(f"val tokens: {len(val_tokens):,}")
    print(f"saved dataset to {output_dir}")


if __name__ == "__main__":
    main()
