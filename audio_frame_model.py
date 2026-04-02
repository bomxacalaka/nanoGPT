import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block, LayerNorm


@dataclass
class AudioFrameGPTConfig:
    block_size: int = 256
    codebook_size: int = 1024
    num_codebooks: int = 8
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class AudioFrameGPT(nn.Module):
    def __init__(self, config: AudioFrameGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.codebook_size, config.n_embd),
                wce=nn.Embedding(config.num_codebooks, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.n_embd, config.codebook_size, bias=False) for _ in range(config.num_codebooks)]
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_frames(self, idx, past_length=0):
        if idx.dim() != 3:
            raise ValueError(f"Expected idx shaped [batch, frames, codebooks], got {tuple(idx.shape)}")
        bsz, frames, codebooks = idx.shape
        if codebooks != self.config.num_codebooks:
            raise ValueError(f"Expected {self.config.num_codebooks} codebooks, got {codebooks}")
        if past_length + frames > self.config.block_size:
            raise ValueError(
                f"past length {past_length} with current length {frames} exceeds block_size {self.config.block_size}"
            )

        tok_emb = self.transformer.wte(idx)
        codebook_positions = torch.arange(self.config.num_codebooks, device=idx.device)
        codebook_emb = self.transformer.wce(codebook_positions).view(1, 1, self.config.num_codebooks, -1)
        frame_emb = (tok_emb + codebook_emb).mean(dim=2)
        pos = torch.arange(past_length, past_length + frames, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        return self.transformer.drop(frame_emb + pos_emb)

    def forward(self, idx, targets=None):
        x = self._embed_frames(idx)
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)

        logits = torch.stack([head(x) for head in self.lm_heads], dim=2)
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.codebook_size),
                targets.reshape(-1),
                ignore_index=-1,
            )
        else:
            logits = logits[:, [-1], :, :]
            loss = None
        return logits, loss

    def forward_with_kv_cache(self, idx, past_kvs=None):
        if past_kvs is None:
            past_kvs = [None] * len(self.transformer.h)
            past_length = 0
        else:
            if len(past_kvs) != len(self.transformer.h):
                raise ValueError("past_kvs length must match transformer depth")
            first_past = next((item for item in past_kvs if item is not None), None)
            past_length = first_past[0].size(2) if first_past is not None else 0

        x = self._embed_frames(idx, past_length=past_length)
        presents = []
        for block, past_kv in zip(self.transformer.h, past_kvs):
            x, present = block(x, past_kv=past_kv, use_cache=True)
            presents.append(present)
        x = self.transformer.ln_f(x)
        logits = torch.stack([head(x[:, [-1], :]) for head in self.lm_heads], dim=2)
        return logits, presents

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
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
def generate_audio_frames(
    model,
    prompt_frames: torch.Tensor,
    *,
    max_new_frames: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    frames = prompt_frames
    cached_logits = None
    cached_kvs = None
    cached_window_length = 0
    if hasattr(model, "forward_with_kv_cache"):
        window = frames[:, -min(frames.size(1), model.config.block_size):, :]
        if window.size(1) > 0:
            cached_logits, cached_kvs = model.forward_with_kv_cache(window)
            cached_window_length = window.size(1)

    for _ in range(max_new_frames):
        if cached_logits is not None:
            logits = cached_logits[:, -1, :, :] / temperature
        else:
            window = frames[:, -model.config.block_size :, :]
            logits, _ = model(window)
            logits = logits[:, -1, :, :] / temperature

        next_codes = []
        for codebook_idx in range(model.config.num_codebooks):
            codebook_logits = logits[:, codebook_idx, :]
            if top_k is not None:
                v, _ = torch.topk(codebook_logits, min(top_k, codebook_logits.size(-1)))
                codebook_logits = codebook_logits.masked_fill(codebook_logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(codebook_logits, dim=-1)
            next_codes.append(torch.multinomial(probs, num_samples=1))
        next_frame = torch.cat(next_codes, dim=1).unsqueeze(1)
        frames = torch.cat((frames, next_frame), dim=1)

        if cached_kvs is not None and cached_window_length < model.config.block_size:
            cached_logits, cached_kvs = model.forward_with_kv_cache(next_frame, past_kvs=cached_kvs)
            cached_window_length += 1
        else:
            window = frames[:, -min(frames.size(1), model.config.block_size) :, :]
            cached_logits, cached_kvs = model.forward_with_kv_cache(window)
            cached_window_length = window.size(1)

    return frames
