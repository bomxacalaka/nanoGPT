from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SemanticFrameUpsamplerConfig:
    n_embd: int = 256
    semantic_to_codec_ratio: int = 3
    dropout: float = 0.0
    bias: bool = False


class SemanticFrameUpsampler(nn.Module):
    """
    Lightweight learnable semantic-to-frame upsampler.

    It first repeats semantic embeddings by the nominal codec ratio, then
    refines them with a small temporal Conv1d stack so frame-level decoder
    conditioning has more local structure than pure nearest-neighbor repeat.
    """

    def __init__(self, config: SemanticFrameUpsamplerConfig):
        super().__init__()
        if config.semantic_to_codec_ratio < 1:
            raise ValueError("semantic_to_codec_ratio must be >= 1")
        self.config = config
        self.refine = nn.Sequential(
            nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1, bias=config.bias),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.n_embd, config.n_embd, kernel_size=3, padding=1, bias=config.bias),
        )

    def forward(self, semantic_emb: torch.Tensor, target_frames: int | None = None) -> torch.Tensor:
        if semantic_emb.dim() != 3:
            raise ValueError(f"Expected semantic_emb shaped [batch, steps, channels], got {tuple(semantic_emb.shape)}")
        upsampled = torch.repeat_interleave(
            semantic_emb,
            repeats=self.config.semantic_to_codec_ratio,
            dim=1,
        )
        if target_frames is not None:
            if upsampled.size(1) < target_frames:
                pad = upsampled[:, -1:, :].expand(-1, target_frames - upsampled.size(1), -1)
                upsampled = torch.cat((upsampled, pad), dim=1)
            elif upsampled.size(1) > target_frames:
                upsampled = upsampled[:, :target_frames, :]

        refined = self.refine(upsampled.transpose(1, 2)).transpose(1, 2)
        return upsampled + refined
