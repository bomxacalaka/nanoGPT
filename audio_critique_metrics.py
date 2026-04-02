from __future__ import annotations

from dataclasses import dataclass

import torch


def _to_list_1d(x: torch.Tensor) -> list[int]:
    if x.dim() != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")
    return x.detach().cpu().to(torch.long).tolist()


def normalized_edit_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a_list = _to_list_1d(a)
    b_list = _to_list_1d(b)
    if not a_list and not b_list:
        return 0.0
    if not a_list:
        return 1.0
    if not b_list:
        return 1.0

    prev = list(range(len(b_list) + 1))
    for i, av in enumerate(a_list, start=1):
        curr = [i]
        for j, bv in enumerate(b_list, start=1):
            cost = 0 if av == bv else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    dist = prev[-1]
    return float(dist) / float(max(len(a_list), len(b_list)))


def token_accuracy(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int | None]:
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError(f"Expected 1D tensors, got {tuple(a.shape)} and {tuple(b.shape)}")
    compare_len = min(a.numel(), b.numel())
    if compare_len == 0:
        return {
            "compare_len": 0,
            "accuracy": None,
            "length_ratio": None,
        }
    a_cmp = a[:compare_len].detach().cpu()
    b_cmp = b[:compare_len].detach().cpu()
    acc = (a_cmp == b_cmp).to(torch.float32).mean().item()
    return {
        "compare_len": int(compare_len),
        "accuracy": float(acc),
        "length_ratio": float(a.numel()) / float(max(1, b.numel())),
    }


def frame_codebook_accuracy(pred_codes: torch.Tensor, target_codes: torch.Tensor) -> dict:
    if pred_codes.dim() != 2 or target_codes.dim() != 2:
        raise ValueError(
            f"Expected frame-code tensors shaped [codebooks, frames], got {tuple(pred_codes.shape)} and {tuple(target_codes.shape)}"
        )
    pred_frames = pred_codes.transpose(0, 1).contiguous()
    target_frames = target_codes.transpose(0, 1).contiguous()
    compare_frames = min(pred_frames.size(0), target_frames.size(0))
    compare_codebooks = min(pred_frames.size(1), target_frames.size(1))
    if compare_frames == 0 or compare_codebooks == 0:
        return {
            "compare_frames": 0,
            "compare_codebooks": 0,
            "overall_accuracy": None,
            "per_codebook_accuracy": [],
        }
    pred_cmp = pred_frames[:compare_frames, :compare_codebooks]
    target_cmp = target_frames[:compare_frames, :compare_codebooks]
    per_codebook = (
        (pred_cmp == target_cmp)
        .to(torch.float32)
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist()
    )
    overall = (pred_cmp == target_cmp).to(torch.float32).mean().item()
    return {
        "compare_frames": int(compare_frames),
        "compare_codebooks": int(compare_codebooks),
        "overall_accuracy": float(overall),
        "per_codebook_accuracy": [float(v) for v in per_codebook],
    }


def prosody_mae(pred_features: torch.Tensor, target_features: torch.Tensor, feature_names: list[str] | tuple[str, ...]) -> dict:
    if pred_features.dim() != 2 or target_features.dim() != 2:
        raise ValueError(
            f"Expected prosody tensors shaped [frames, features], got {tuple(pred_features.shape)} and {tuple(target_features.shape)}"
        )
    compare_frames = min(pred_features.size(0), target_features.size(0))
    compare_dims = min(pred_features.size(1), target_features.size(1), len(feature_names))
    if compare_frames == 0 or compare_dims == 0:
        return {
            "compare_frames": 0,
            "mean_abs_error": None,
            "per_feature_mae": {},
        }
    pred_cmp = pred_features[:compare_frames, :compare_dims].detach().cpu()
    target_cmp = target_features[:compare_frames, :compare_dims].detach().cpu()
    mae_vec = (pred_cmp - target_cmp).abs().mean(dim=0)
    return {
        "compare_frames": int(compare_frames),
        "mean_abs_error": float(mae_vec.mean().item()),
        "per_feature_mae": {
            str(feature_names[i]): float(mae_vec[i].item())
            for i in range(compare_dims)
        },
    }
