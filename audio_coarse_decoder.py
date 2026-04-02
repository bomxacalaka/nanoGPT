import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from audio_semantic_upsampler import SemanticFrameUpsampler
from audio_semantic_upsampler import SemanticFrameUpsamplerConfig
from model import LayerNorm


@dataclass
class AudioCoarseDecoderConfig:
    block_size: int = 384
    semantic_vocab_size: int = 1024
    codec_vocab_size: int = 1024
    semantic_to_codec_ratio: int = 3
    style_prompt_frames: int = 96
    prosody_dim: int = 3
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class AudioCoarseDecoder(nn.Module):
    def __init__(self, config: AudioCoarseDecoderConfig):
        super().__init__()
        if config.semantic_to_codec_ratio < 1:
            raise ValueError("semantic_to_codec_ratio must be >= 1")
        self.config = config
        self.memory_block_size = config.block_size + config.style_prompt_frames

        self.semantic_embed = nn.Embedding(config.semantic_vocab_size, config.n_embd)
        self.coarse_embed = nn.Embedding(config.codec_vocab_size, config.n_embd)
        self.prompt_codec_embed = nn.Embedding(config.codec_vocab_size, config.n_embd)
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

        self.memory_pos_embed = nn.Embedding(self.memory_block_size, config.n_embd)
        self.target_pos_embed = nn.Embedding(config.block_size + 1, config.n_embd)
        self.decoder_bos = nn.Parameter(torch.zeros(1, 1, config.n_embd))
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
        self.lm_head = nn.Linear(config.n_embd, config.codec_vocab_size, bias=False)

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
            coarse_prompt = prompt_codec_frames[:, : self.config.style_prompt_frames, 0]
            prompt_emb = self.prompt_codec_embed(coarse_prompt)
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

    def build_decoder_inputs(self, coarse_input_tokens: torch.Tensor | None, style_vector: torch.Tensor | None):
        if coarse_input_tokens is None:
            batch_size = style_vector.size(0) if style_vector is not None else 1
            tgt = self.decoder_bos.expand(batch_size, 1, -1)
        else:
            coarse_emb = self.coarse_embed(coarse_input_tokens)
            bos = self.decoder_bos.expand(coarse_input_tokens.size(0), 1, -1)
            tgt = torch.cat((bos, coarse_emb), dim=1)
        if tgt.size(1) > self.config.block_size + 1:
            tgt = tgt[:, -(self.config.block_size + 1) :, :]
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
        *,
        target_frames: int,
        prompt_codec_frames: torch.Tensor | None = None,
        prompt_prosody_features: torch.Tensor | None = None,
        prompt_valid_lengths: torch.Tensor | None = None,
        coarse_input_tokens: torch.Tensor | None = None,
    ):
        memory, style_vector, memory_mask = self.encode_memory(
            semantic_tokens,
            target_frames=target_frames,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
        )
        tgt = self.build_decoder_inputs(coarse_input_tokens, style_vector=style_vector)
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=tgt.device),
            diagonal=1,
        )
        hidden = self.decoder(
            tgt,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_mask,
        )
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)

    def forward(
        self,
        semantic_tokens,
        prompt_codec_frames=None,
        prompt_prosody_features=None,
        prompt_valid_lengths=None,
        coarse_targets=None,
        target_frames=None,
    ):
        if coarse_targets is None and target_frames is None:
            raise ValueError("Either coarse_targets or target_frames must be provided")

        if coarse_targets is None:
            logits = self.decode_logits(
                semantic_tokens,
                target_frames=target_frames,
                prompt_codec_frames=prompt_codec_frames,
                prompt_prosody_features=prompt_prosody_features,
                prompt_valid_lengths=prompt_valid_lengths,
                coarse_input_tokens=None,
            )
            return logits, None

        decoder_input_tokens = coarse_targets[:, :-1] if coarse_targets.size(1) > 1 else None
        logits = self.decode_logits(
            semantic_tokens,
            target_frames=coarse_targets.size(1),
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            coarse_input_tokens=decoder_input_tokens,
        )
        logits = logits[:, -coarse_targets.size(1):, :]
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.codec_vocab_size),
            coarse_targets.reshape(-1),
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
def decode_coarse_codes_from_semantics(
    model,
    semantic_tokens: torch.Tensor,
    *,
    prompt_codec_frames: torch.Tensor | None = None,
    prompt_prosody_features: torch.Tensor | None = None,
    prompt_valid_lengths: torch.Tensor | None = None,
    target_frames: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    full_generated = None
    current = None
    for _ in range(target_frames):
        logits = model.decode_logits(
            semantic_tokens,
            target_frames=target_frames,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            coarse_input_tokens=current,
        )
        next_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        full_generated = next_token if full_generated is None else torch.cat((full_generated, next_token), dim=1)
        current = next_token if current is None else torch.cat((current, next_token), dim=1)
        if current.size(1) > model.config.block_size:
            current = current[:, -model.config.block_size :]
    return full_generated
