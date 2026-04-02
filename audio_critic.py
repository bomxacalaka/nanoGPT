from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchaudio


@dataclass
class AudioCriticConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    max_frames: int = 512
    base_channels: int = 64
    hidden_dim: int = 256
    dropout: float = 0.1


class AudioCritic(nn.Module):
    """
    Lightweight multi-head speech critic.

    Inputs:
    - prompt waveform
    - generated continuation waveform
    - target continuation waveform

    Outputs:
    - realism_score
    - intelligibility_score
    - style_match_score
    - prosody_match_score
    - semantic_match_score
    """

    def __init__(self, config: AudioCriticConfig):
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            n_mels=config.n_mels,
            power=2.0,
        )

        in_ch = config.n_mels
        c = config.base_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, c, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(c, c * 2, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(c * 2, c * 4, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        feat_dim = c * 4
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 5, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "realism": nn.Linear(config.hidden_dim, 1),
                "intelligibility": nn.Linear(config.hidden_dim, 1),
                "style_match": nn.Linear(config.hidden_dim, 1),
                "prosody_match": nn.Linear(config.hidden_dim, 1),
                "semantic_match": nn.Linear(config.hidden_dim, 1),
            }
        )

    def _ensure_mono(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() == 2:
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav
        if wav.dim() == 3:
            if wav.size(1) > 1:
                wav = wav.mean(dim=1, keepdim=True)
            return wav
        raise ValueError(f"Expected waveform shaped [channels, samples] or [batch, channels, samples], got {tuple(wav.shape)}")

    def _crop_or_pad(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.size(-1) < self.config.max_frames:
            pad = self.config.max_frames - feat.size(-1)
            feat = torch.nn.functional.pad(feat, (0, pad))
        elif feat.size(-1) > self.config.max_frames:
            feat = feat[..., : self.config.max_frames]
        return feat

    def encode_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self._ensure_mono(wav)
        mel = self.mel(wav).clamp_min(1e-5).log()
        if mel.dim() == 4 and mel.size(1) == 1:
            mel = mel.squeeze(1)
        mel = self._crop_or_pad(mel)
        hidden = self.encoder(mel).squeeze(-1)
        return hidden

    def forward(self, prompt_wav: torch.Tensor, generated_wav: torch.Tensor, target_wav: torch.Tensor) -> dict[str, torch.Tensor]:
        prompt_feat = self.encode_waveform(prompt_wav)
        generated_feat = self.encode_waveform(generated_wav)
        target_feat = self.encode_waveform(target_wav)
        diff_gt = (generated_feat - target_feat).abs()
        diff_pg = (prompt_feat - generated_feat).abs()
        joint = torch.cat((prompt_feat, generated_feat, target_feat, diff_gt, diff_pg), dim=-1)
        hidden = self.proj(joint)
        return {name: torch.sigmoid(head(hidden)).squeeze(-1) for name, head in self.heads.items()}
