"""
Build K-means centroids over HuBERT features for the v5 semantic tokenizer.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import torchaudio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from audio_codec import load_audio


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--bundle_name", default="HUBERT_BASE")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--num_clusters", type=int, default=256)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=50000)
    parser.add_argument("--kmeans_iters", type=int, default=40)
    parser.add_argument("--assign_batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
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


def progress_iter(files, desc: str):
    if tqdm is not None:
        return tqdm(files, desc=desc, unit="file", file=sys.stdout, dynamic_ncols=False, leave=True)
    return files


def extract_hubert_features(model, sample_rate, path: Path, device: str, layer: int, frame_stride: int):
    wav, wav_sr = load_audio(str(path))
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if wav_sr != sample_rate:
        wav = torchaudio.functional.resample(wav, wav_sr, sample_rate)
    with torch.no_grad():
        feature_layers, _ = model.extract_features(wav.to(device))
    layer_index = layer if layer >= 0 else len(feature_layers) + layer
    features = feature_layers[layer_index]
    if frame_stride > 1:
        features = features[:, ::frame_stride, :]
    return features.squeeze(0).float().cpu()


def reservoir_update(rng: random.Random, reservoir, vector, seen_count, max_frames):
    if len(reservoir) < max_frames:
        reservoir.append(vector.clone())
        return
    replace_idx = rng.randrange(seen_count)
    if replace_idx < max_frames:
        reservoir[replace_idx] = vector.clone()


def sample_feature_bank(files, model, sample_rate, *, device: str, layer: int, frame_stride: int, max_frames: int, seed: int):
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for idx, path in enumerate(progress_iter(files, "hubert"), start=1):
        features = extract_hubert_features(model, sample_rate, path, device, layer, frame_stride)
        for row in features:
            seen += 1
            reservoir_update(rng, reservoir, row, seen, max_frames)
        if tqdm is None:
            print(f"[hubert] {idx}/{len(files)} {path} frames={features.size(0)} seen={seen} kept={len(reservoir)}", flush=True)
    if not reservoir:
        raise SystemExit("No HuBERT features were extracted; check the input_dir and audio files.")
    bank = torch.stack(reservoir, dim=0)
    return bank


def pairwise_distances(x: torch.Tensor, centers: torch.Tensor):
    x_norm = (x * x).sum(dim=1, keepdim=True)
    c_norm = (centers * centers).sum(dim=1).unsqueeze(0)
    return x_norm + c_norm - 2.0 * (x @ centers.t())


def kmeans(features: torch.Tensor, num_clusters: int, *, iters: int, assign_batch_size: int, seed: int):
    if features.size(0) < num_clusters:
        raise ValueError(
            f"Need at least num_clusters={num_clusters} feature rows, got {features.size(0)}"
        )
    generator_device = features.device.type if features.is_cuda else "cpu"
    g = torch.Generator(device=generator_device)
    g.manual_seed(seed)
    init_idx = torch.randperm(features.size(0), generator=g, device=features.device)[:num_clusters]
    centers = features[init_idx].clone()

    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, device=features.device, dtype=torch.long)
        for start in range(0, features.size(0), assign_batch_size):
            batch = features[start:start + assign_batch_size]
            distances = pairwise_distances(batch, centers)
            assign = torch.argmin(distances, dim=1)
            counts += torch.bincount(assign, minlength=num_clusters)
            sums.index_add_(0, assign, batch)
        nonzero = counts > 0
        centers[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(1).to(features.dtype)
        if (~nonzero).any():
            refill_idx = torch.randperm(
                features.size(0),
                generator=g,
                device=features.device,
            )[: int((~nonzero).sum().item())]
            centers[~nonzero] = features[refill_idx]
    return centers


def main():
    args = parse_args()
    files = iter_audio_files(args.input_dir, args.max_files)
    if not files:
        raise SystemExit(f"No audio files found under {args.input_dir}")

    try:
        bundle = getattr(torchaudio.pipelines, args.bundle_name)
    except AttributeError as exc:
        raise SystemExit(f"Unsupported torchaudio pipeline bundle: {args.bundle_name}") from exc

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = bundle.get_model().to(args.device)
    model.eval()
    feature_bank = sample_feature_bank(
        files,
        model,
        bundle.sample_rate,
        device=args.device,
        layer=args.layer,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        seed=args.seed,
    ).to(args.device)

    print(
        f"Collected {feature_bank.size(0)} HuBERT frames with dim={feature_bank.size(1)}; "
        f"running K-means with k={args.num_clusters}",
        flush=True,
    )
    centers = kmeans(
        feature_bank,
        args.num_clusters,
        iters=args.kmeans_iters,
        assign_batch_size=args.assign_batch_size,
        seed=args.seed,
    ).cpu().numpy().astype(np.float32, copy=False)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, centers)
    print(f"Saved centroids to {output_path}", flush=True)


if __name__ == "__main__":
    main()
