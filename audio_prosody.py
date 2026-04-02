from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchaudio


def _resample_feature_1d(values: torch.Tensor, target_frames: int) -> torch.Tensor:
    if values.numel() == target_frames:
        return values
    if values.numel() <= 1:
        return values.new_full((target_frames,), float(values[0]) if values.numel() == 1 else 0.0)
    values = values.view(1, 1, -1)
    resized = F.interpolate(values, size=target_frames, mode="linear", align_corners=False)
    return resized.view(-1)


def extract_prosody_features(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    target_frames: int,
    frame_rate_hz: float = 75.0,
) -> torch.Tensor:
    """
    Return prompt prosody features shaped [target_frames, 3]:
    [log_pitch_hz, log_energy, voiced]
    """
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform shaped [channels, samples], got {tuple(wav.shape)}")
    if target_frames <= 0:
        raise ValueError("target_frames must be > 0")

    mono = wav.mean(dim=0, keepdim=True)
    frame_time = 1.0 / float(frame_rate_hz)

    pitch_hz = torchaudio.functional.detect_pitch_frequency(
        mono,
        sample_rate=sample_rate,
        frame_time=frame_time,
    ).squeeze(0)
    voiced = (pitch_hz > 1.0).to(torch.float32)
    pitch_feature = torch.log1p(torch.clamp(pitch_hz, min=0.0))

    hop_length = max(1, int(round(sample_rate * frame_time)))
    win_length = max(hop_length, int(round(sample_rate * frame_time * 2)))
    if mono.size(-1) < win_length:
        pad = win_length - mono.size(-1)
        mono_for_energy = F.pad(mono, (0, pad))
    else:
        mono_for_energy = mono
    frames = mono_for_energy.unfold(-1, win_length, hop_length)
    energy = frames.pow(2).mean(dim=-1).sqrt().squeeze(0)
    energy_feature = torch.log1p(torch.clamp(energy, min=0.0))

    pitch_feature = _resample_feature_1d(pitch_feature, target_frames)
    energy_feature = _resample_feature_1d(energy_feature, target_frames)
    voiced = _resample_feature_1d(voiced, target_frames)

    return torch.stack((pitch_feature, energy_feature, voiced), dim=-1).to(torch.float32)
