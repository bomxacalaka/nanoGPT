"""
Prepare aligned semantic-token and acoustic-codec datasets for V5 speech modeling.

Outputs:
- train.semantic.bin / val.semantic.bin
- train.codec.bin / val.codec.bin
- train.align.npy / val.align.npy
- meta.pkl

The alignment file stores per-utterance rows shaped:
    [semantic_offset, semantic_length, codec_frame_offset, codec_frame_length]

The codec bin stores frame-major raw codec ids flattened as:
    frame0_codebook0, frame0_codebook1, ..., frame1_codebook0, ...
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from audio_codec import EncodecUnavailableError
from audio_codec import encode_waveform
from audio_codec import frame_codes_to_flat_tokens
from audio_codec import load_audio
from audio_codec import load_encodec_model
from audio_prosody import extract_prosody_features
from audio_semantic_tokenizer import EncodecCoarseSemanticBackend
from audio_semantic_tokenizer import HubertKmeansSemanticBackend
from audio_semantic_tokenizer import build_default_semantic_tokenizer


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing source audio files.")
    parser.add_argument("--val_input_dir", default="", help="Optional separate validation audio directory.")
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for semantic/acoustic bins and metadata.",
    )
    parser.add_argument("--model_name", default="encodec_24khz", choices=["encodec_24khz"])
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument(
        "--semantic_backend",
        default=EncodecCoarseSemanticBackend.backend_name,
        choices=[EncodecCoarseSemanticBackend.backend_name, HubertKmeansSemanticBackend.backend_name],
    )
    parser.add_argument("--semantic_codebook_index", type=int, default=0)
    parser.add_argument("--semantic_frame_stride", type=int, default=3)
    parser.add_argument("--hubert_bundle_name", default="HUBERT_BASE")
    parser.add_argument("--hubert_centroids_path", default="")
    parser.add_argument("--hubert_layer", type=int, default=-1)
    parser.add_argument("--hubert_frame_stride", type=int, default=2)
    parser.add_argument("--codec_device", default="cpu")
    parser.add_argument("--semantic_device", default="cpu")
    parser.add_argument("--val_frac", type=float, default=0.01)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
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


def progress_iter(files, split_name: str):
    if tqdm is not None:
        return tqdm(files, desc=f"{split_name:5s}", unit="file", file=sys.stdout, dynamic_ncols=False, leave=True)
    return files


def dtype_for_vocab(vocab_size: int):
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def encode_split(files, split_name: str, semantic_tokenizer, codec_model, args):
    semantic_chunks = []
    codec_chunks = []
    prosody_chunks = []
    align_rows = []
    semantic_rates = []
    semantic_offset = 0
    codec_frame_offset = 0
    observed_num_codebooks = None
    observed_semantic_vocab_size = None

    semantic_token_dtype = None
    codec_token_dtype = dtype_for_vocab(args.codebook_size)

    for index, path in enumerate(progress_iter(files, split_name), start=1):
        wav, sample_rate = load_audio(str(path))
        semantic_batch = semantic_tokenizer.encode_waveform(wav, sample_rate)
        codec_batch = encode_waveform(
            codec_model,
            wav,
            sample_rate,
            device=args.codec_device,
            codebook_size=args.codebook_size,
        )
        observed_num_codebooks = codec_batch.num_codebooks
        codec_frames = codec_batch.codes.transpose(0, 1).contiguous()
        semantic_rates.append(float(semantic_batch.semantic_rate_hz))
        observed_semantic_vocab_size = semantic_batch.vocab_size

        if semantic_token_dtype is None:
            semantic_token_dtype = dtype_for_vocab(int(semantic_batch.vocab_size))
        semantic_np = semantic_batch.tokens.cpu().numpy().astype(semantic_token_dtype, copy=False)
        codec_np = codec_frames.cpu().numpy().astype(codec_token_dtype, copy=False)
        codec_flat_np = frame_codes_to_flat_tokens(torch_from_numpy(codec_np), args.codebook_size).cpu().numpy().astype(
            codec_token_dtype,
            copy=False,
        )
        prosody_np = extract_prosody_features(
            semantic_batch.normalized_wav if semantic_batch.normalized_wav is not None else wav,
            codec_batch.sample_rate,
            target_frames=int(codec_np.shape[0]),
            frame_rate_hz=75.0,
        ).cpu().numpy().astype(np.float32, copy=False)

        semantic_chunks.append(semantic_np)
        codec_chunks.append(codec_flat_np)
        prosody_chunks.append(prosody_np)
        align_rows.append(
            [
                semantic_offset,
                int(semantic_np.shape[0]),
                codec_frame_offset,
                int(codec_np.shape[0]),
            ]
        )
        semantic_offset += int(semantic_np.shape[0])
        codec_frame_offset += int(codec_np.shape[0])

        if tqdm is None:
            print(
                f"[{split_name}] {index}/{len(files)} {path} "
                f"semantic={semantic_np.shape[0]} codec_frames={codec_np.shape[0]}",
                flush=True,
            )

    if semantic_chunks:
        semantic_flat = np.concatenate(semantic_chunks)
    else:
        semantic_flat = np.array([], dtype=semantic_token_dtype or np.uint16)
    if codec_chunks:
        codec_flat = np.concatenate(codec_chunks)
    else:
        codec_flat = np.array([], dtype=codec_token_dtype)
    if prosody_chunks:
        prosody = np.concatenate(prosody_chunks, axis=0)
    else:
        prosody = np.zeros((0, 3), dtype=np.float32)
    align = np.asarray(align_rows, dtype=np.int64)
    semantic_rate_hz = float(np.mean(semantic_rates)) if semantic_rates else 0.0
    return (
        semantic_flat,
        codec_flat,
        prosody,
        align,
        observed_num_codebooks,
        observed_semantic_vocab_size,
        semantic_rate_hz,
    )


def torch_from_numpy(array: np.ndarray):
    import torch

    return torch.from_numpy(array.astype(np.int64, copy=False))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        codec_model = load_encodec_model(
            model_name=args.model_name,
            bandwidth=args.bandwidth,
            device=args.codec_device,
        )
    except EncodecUnavailableError as exc:
        raise SystemExit(str(exc))

    semantic_tokenizer = build_default_semantic_tokenizer(
        semantic_backend=args.semantic_backend,
        device=args.semantic_device,
        frame_stride=args.semantic_frame_stride,
        semantic_codebook_index=args.semantic_codebook_index,
        model_name=args.model_name,
        bandwidth=args.bandwidth,
        codebook_size=args.codebook_size,
        hubert_bundle_name=args.hubert_bundle_name,
        hubert_centroids_path=args.hubert_centroids_path,
        hubert_layer=args.hubert_layer,
        hubert_frame_stride=args.hubert_frame_stride,
    )

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

    (
        train_semantic,
        train_codec,
        train_prosody,
        train_align,
        train_num_codebooks,
        train_semantic_vocab_size,
        train_semantic_rate_hz,
    ) = encode_split(
        train_files,
        "train",
        semantic_tokenizer,
        codec_model,
        args,
    )
    (
        val_semantic,
        val_codec,
        val_prosody,
        val_align,
        val_num_codebooks,
        val_semantic_vocab_size,
        val_semantic_rate_hz,
    ) = encode_split(
        val_files,
        "val",
        semantic_tokenizer,
        codec_model,
        args,
    )
    num_codebooks = train_num_codebooks or val_num_codebooks
    semantic_vocab_size = train_semantic_vocab_size or val_semantic_vocab_size or args.codebook_size
    semantic_rate_hz = train_semantic_rate_hz or val_semantic_rate_hz
    semantic_to_codec_ratio = max(1, int(round(75.0 / max(semantic_rate_hz, 1e-8))))

    (output_dir / "train.semantic.bin").write_bytes(train_semantic.tobytes())
    (output_dir / "val.semantic.bin").write_bytes(val_semantic.tobytes())
    (output_dir / "train.codec.bin").write_bytes(train_codec.tobytes())
    (output_dir / "val.codec.bin").write_bytes(val_codec.tobytes())
    np.save(output_dir / "train.prosody.npy", train_prosody)
    np.save(output_dir / "val.prosody.npy", val_prosody)
    np.save(output_dir / "train.align.npy", train_align)
    np.save(output_dir / "val.align.npy", val_align)

    meta = {
        "dataset_type": "audio_semantic_codec_v5",
        "semantic_tokenizer": semantic_tokenizer.to_metadata(),
        "semantic_vocab_size": int(semantic_vocab_size),
        "semantic_to_codec_ratio": int(semantic_to_codec_ratio),
        "codec_model_name": args.model_name,
        "bandwidth": args.bandwidth,
        "codebook_size": args.codebook_size,
        "codec_vocab_size": args.codebook_size,
        "semantic_dtype": np.dtype(train_semantic.dtype).name,
        "codec_dtype": np.dtype(train_codec.dtype).name,
        "num_codebooks": num_codebooks,
        "sample_rate": codec_model.sample_rate,
        "channels": codec_model.channels,
        "semantic_rate_hz_estimate": float(semantic_rate_hz),
        "codec_frame_rate_hz_estimate": 75.0,
        "alignment_strategy": "utterance_aligned_semantic_and_codec_streams",
        "alignment_fields": [
            "semantic_offset",
            "semantic_length",
            "codec_frame_offset",
            "codec_frame_length",
        ],
        "prosody_feature_names": ["log_pitch_hz", "log_energy", "voiced"],
        "train_num_utterances": int(train_align.shape[0]),
        "val_num_utterances": int(val_align.shape[0]),
        "train_num_semantic_tokens": int(train_semantic.shape[0]),
        "val_num_semantic_tokens": int(val_semantic.shape[0]),
        "train_num_codec_frames": int(train_align[:, 3].sum()) if train_align.size else 0,
        "val_num_codec_frames": int(val_align[:, 3].sum()) if val_align.size else 0,
        "train_num_prosody_frames": int(train_prosody.shape[0]),
        "val_num_prosody_frames": int(val_prosody.shape[0]),
        "token_dtype": np.dtype(dtype_for_vocab(args.codebook_size)).name,
    }

    with (output_dir / "meta.pkl").open("wb") as handle:
        pickle.dump(meta, handle)

    print(f"Wrote semantic/acoustic dataset to {output_dir}")
    print(
        f"Train semantic tokens: {meta['train_num_semantic_tokens']:,} | "
        f"Train codec frames: {meta['train_num_codec_frames']:,}"
    )
    print(
        f"Val semantic tokens: {meta['val_num_semantic_tokens']:,} | "
        f"Val codec frames: {meta['val_num_codec_frames']:,}"
    )


if __name__ == "__main__":
    main()
