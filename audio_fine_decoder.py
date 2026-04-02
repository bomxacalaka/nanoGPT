import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from audio_semantic_upsampler import SemanticFrameUpsampler
from audio_semantic_upsampler import SemanticFrameUpsamplerConfig
from model import LayerNorm


@dataclass
class AudioFineDecoderConfig:
    block_size: int = 384
    semantic_vocab_size: int = 1024
    codec_vocab_size: int = 1024
    num_codebooks: int = 8
    semantic_to_codec_ratio: int = 3
    style_prompt_frames: int = 96
    prosody_dim: int = 3
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class AudioFineDecoder(nn.Module):
    """
    Predict residual codec codebooks [1..N-1] from:
    - semantic tokens
    - coarse codebook 0 sequence
    - prompt codec prefix
    - prompt prosody

    This stage is intentionally non-autoregressive over residual codebooks so it
    can render detail in one pass after the coarse structure exists.
    """

    def __init__(self, config: AudioFineDecoderConfig):
        super().__init__()
        if config.num_codebooks < 2:
            raise ValueError("AudioFineDecoder requires num_codebooks >= 2")
        if config.semantic_to_codec_ratio < 1:
            raise ValueError("semantic_to_codec_ratio must be >= 1")
        self.config = config
        self.num_residual_codebooks = config.num_codebooks - 1
        self.memory_block_size = config.block_size + config.style_prompt_frames

        self.semantic_embed = nn.Embedding(config.semantic_vocab_size, config.n_embd)
        self.coarse_embed = nn.Embedding(config.codec_vocab_size, config.n_embd)
        self.prompt_codec_embed = nn.Embedding(config.codec_vocab_size, config.n_embd)
        self.codebook_embed = nn.Embedding(config.num_codebooks, config.n_embd)
        self.prosody_proj = nn.Linear(config.prosody_dim, config.n_embd, bias=config.bias)
        self.upsampler = SemanticFrameUpsampler(
            SemanticFrameUpsamplerConfig(
                n_embd=config.n_embd,
                semantic_to_codec_ratio=config.semantic_to_codec_ratio,
                dropout=config.dropout,
                bias=config.bias,
            )
        )

        self.semantic_modality = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.prompt_modality = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.coarse_modality = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        self.memory_pos_embed = nn.Embedding(self.memory_block_size, config.n_embd)
        self.target_pos_embed = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=config.bias,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            bias=config.bias,
        )
        self.memory_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layer)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.style_score = nn.Linear(config.n_embd, 1, bias=config.bias)
        self.style_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.style_gate = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.codec_heads = nn.ModuleList(
            [nn.Linear(config.n_embd, config.codec_vocab_size, bias=False) for _ in range(self.num_residual_codebooks)]
        )

        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.memory_pos_embed.weight.numel()
            n_params -= self.target_pos_embed.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_prompt_frames(
        self,
        prompt_codec_frames: torch.Tensor | None,
        prompt_prosody_features: torch.Tensor | None,
        prompt_valid_lengths: torch.Tensor | None = None,
    ):
        if prompt_codec_frames is None and prompt_prosody_features is None:
            return None, None

        prompt_emb = None
        frame_count = None
        if prompt_codec_frames is not None:
            if prompt_codec_frames.dim() != 3:
                raise ValueError(
                    f"Expected prompt_codec_frames shaped [batch, frames, codebooks], got {tuple(prompt_codec_frames.shape)}"
                )
            prompt_codec_frames = prompt_codec_frames[:, : self.config.style_prompt_frames, :]
            tok_emb = self.prompt_codec_embed(prompt_codec_frames)
            codebook_pos = torch.arange(self.config.num_codebooks, device=prompt_codec_frames.device)
            codebook_emb = self.codebook_embed(codebook_pos).view(1, 1, self.config.num_codebooks, -1)
            prompt_emb = (tok_emb + codebook_emb).mean(dim=2)
            frame_count = prompt_emb.size(1)

        if prompt_prosody_features is not None:
            prompt_prosody_features = prompt_prosody_features[:, : self.config.style_prompt_frames, :]
            prosody_emb = self.prosody_proj(prompt_prosody_features)
            if prompt_emb is None:
                prompt_emb = prosody_emb
            else:
                min_frames = min(prompt_emb.size(1), prosody_emb.size(1))
                prompt_emb = prompt_emb[:, :min_frames, :] + prosody_emb[:, :min_frames, :]
                frame_count = min_frames
        elif prompt_emb is not None:
            frame_count = prompt_emb.size(1)

        prompt_padding_mask = None
        if prompt_emb is not None and prompt_valid_lengths is not None:
            clipped_lengths = prompt_valid_lengths.to(device=prompt_emb.device).clamp(min=0, max=frame_count)
            frame_pos = torch.arange(frame_count, device=prompt_emb.device).unsqueeze(0)
            prompt_padding_mask = frame_pos >= clipped_lengths.unsqueeze(1)
            prompt_emb = prompt_emb.masked_fill(prompt_padding_mask.unsqueeze(-1), 0.0)

        return prompt_emb, prompt_padding_mask

    def encode_memory(
        self,
        semantic_tokens: torch.Tensor,
        *,
        target_frames: int,
        prompt_codec_frames: torch.Tensor | None = None,
        prompt_prosody_features: torch.Tensor | None = None,
        prompt_valid_lengths: torch.Tensor | None = None,
    ):
        semantic_emb = self.semantic_embed(semantic_tokens)
        semantic_frames = self.upsampler(semantic_emb, target_frames=target_frames) + self.semantic_modality

        memory_parts = [semantic_frames]
        mask_parts = [
            torch.zeros(
                semantic_frames.size(0),
                semantic_frames.size(1),
                dtype=torch.bool,
                device=semantic_frames.device,
            )
        ]

        prompt_emb, prompt_padding_mask = self._embed_prompt_frames(
            prompt_codec_frames,
            prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
        )
        if prompt_emb is not None:
            memory_parts.append(prompt_emb + self.prompt_modality)
            if prompt_padding_mask is None:
                prompt_padding_mask = torch.zeros(
                    prompt_emb.size(0),
                    prompt_emb.size(1),
                    dtype=torch.bool,
                    device=prompt_emb.device,
                )
            mask_parts.append(prompt_padding_mask)

        memory = torch.cat(memory_parts, dim=1)
        memory_mask = torch.cat(mask_parts, dim=1)
        if memory.size(1) > self.memory_block_size:
            memory = memory[:, -self.memory_block_size :, :]
            memory_mask = memory_mask[:, -self.memory_block_size :]

        pos = torch.arange(memory.size(1), dtype=torch.long, device=memory.device)
        memory = self.drop(memory + self.memory_pos_embed(pos))
        memory = self.memory_encoder(memory, src_key_padding_mask=memory_mask)

        style_vector = None
        if prompt_emb is not None:
            style_logits = self.style_score(torch.tanh(prompt_emb)).squeeze(-1)
            if prompt_padding_mask is not None:
                style_logits = style_logits.masked_fill(prompt_padding_mask, float("-inf"))
            attn = torch.softmax(style_logits, dim=1)
            if prompt_padding_mask is not None:
                attn = attn.masked_fill(prompt_padding_mask, 0.0)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
            style_vector = torch.sum(prompt_emb * attn.unsqueeze(-1), dim=1)
        return memory, style_vector, memory_mask

    def build_target_inputs(self, coarse_tokens: torch.Tensor, style_vector: torch.Tensor | None):
        if coarse_tokens.dim() != 2:
            raise ValueError(f"Expected coarse_tokens shaped [batch, frames], got {tuple(coarse_tokens.shape)}")
        tgt = self.coarse_embed(coarse_tokens) + self.coarse_modality
        if tgt.size(1) > self.config.block_size:
            tgt = tgt[:, -self.config.block_size :, :]
        pos = torch.arange(tgt.size(1), dtype=torch.long, device=tgt.device)
        tgt = self.drop(tgt + self.target_pos_embed(pos))
        if style_vector is not None:
            style_add = self.style_proj(style_vector).unsqueeze(1)
            style_gate = torch.sigmoid(self.style_gate(style_vector)).unsqueeze(1)
            tgt = (tgt * (1.0 + style_gate)) + style_add
        return tgt

    def decode_logits(
        self,
        semantic_tokens: torch.Tensor,
        coarse_tokens: torch.Tensor,
        *,
        prompt_codec_frames: torch.Tensor | None = None,
        prompt_prosody_features: torch.Tensor | None = None,
        prompt_valid_lengths: torch.Tensor | None = None,
    ):
        target_frames = coarse_tokens.size(1)
        memory, style_vector, memory_mask = self.encode_memory(
            semantic_tokens,
            target_frames=target_frames,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
        )
        tgt = self.build_target_inputs(coarse_tokens, style_vector=style_vector)
        hidden = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=memory_mask,
        )
        hidden = self.ln_f(hidden)
        return torch.stack([head(hidden) for head in self.codec_heads], dim=2)

    def forward(
        self,
        semantic_tokens,
        coarse_tokens,
        prompt_codec_frames=None,
        prompt_prosody_features=None,
        prompt_valid_lengths=None,
        residual_targets=None,
    ):
        logits = self.decode_logits(
            semantic_tokens,
            coarse_tokens,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
        )
        if residual_targets is None:
            return logits, None
        if residual_targets.dim() != 3:
            raise ValueError(
                f"Expected residual_targets shaped [batch, frames, residual_codebooks], got {tuple(residual_targets.shape)}"
            )
        logits = logits[:, -residual_targets.size(1):, :, :]
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.codec_vocab_size),
            residual_targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **({"fused": True} if use_fused else {}),
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        n_params = self.get_num_params()
        cfg = self.config
        layers, heads, q_width, frames = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_frame = 6 * n_params + 12 * layers * heads * q_width * frames
        flops_per_fwdbwd = flops_per_frame * frames
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised


@torch.no_grad()
def decode_fine_codes_from_semantics_and_coarse(
    model: AudioFineDecoder,
    semantic_tokens: torch.Tensor,
    coarse_tokens: torch.Tensor,
    *,
    prompt_codec_frames: torch.Tensor | None = None,
    prompt_prosody_features: torch.Tensor | None = None,
    prompt_valid_lengths: torch.Tensor | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    ratio = int(model.config.semantic_to_codec_ratio)
    chunk_size = int(model.config.block_size)
    outputs = []
    total_frames = coarse_tokens.size(1)
    for frame_start in range(0, total_frames, chunk_size):
        frame_end = min(total_frames, frame_start + chunk_size)
        coarse_chunk = coarse_tokens[:, frame_start:frame_end]
        semantic_start = min(semantic_tokens.size(1) - 1, frame_start // ratio)
        semantic_end = max(semantic_start + 1, min(semantic_tokens.size(1), (frame_end + ratio - 1) // ratio))
        semantic_chunk = semantic_tokens[:, semantic_start:semantic_end]
        logits, _ = model(
            semantic_chunk,
            coarse_chunk,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            residual_targets=None,
        )
        logits = logits[:, -coarse_chunk.size(1):, :, :] / temperature
        if top_k is not None:
            applied_top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, applied_top_k, dim=-1)
            logits = logits.masked_fill(logits < values[..., [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        samples = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1)
        outputs.append(samples.view(probs.size(0), probs.size(1), probs.size(2)))
    return torch.cat(outputs, dim=1)
