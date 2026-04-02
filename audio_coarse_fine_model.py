import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block, LayerNorm


@dataclass
class AudioCoarseFineGPTConfig:
    block_size: int = 256
    codebook_size: int = 1024
    num_codebooks: int = 8
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    fine_loss_weight: float = 1.0


class AudioCoarseFineGPT(nn.Module):
    def __init__(self, config: AudioCoarseFineGPTConfig):
        super().__init__()
        self.config = config
        if config.num_codebooks < 2:
            raise ValueError("AudioCoarseFineGPT requires at least 2 codebooks")

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.codebook_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.coarse_head = nn.Linear(config.n_embd, config.codebook_size, bias=False)
        self.fine_codebook_emb = nn.Embedding(config.num_codebooks - 1, config.n_embd)
        self.fine_proj = nn.Linear(config.n_embd * 2, config.n_embd, bias=config.bias)
        self.fine_heads = nn.ModuleList(
            [nn.Linear(config.n_embd, config.codebook_size, bias=False) for _ in range(config.num_codebooks - 1)]
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

    def _coarse_input(self, idx):
        if idx.dim() == 3:
            return idx[:, :, 0]
        if idx.dim() == 2:
            return idx
        raise ValueError(f"Expected idx shaped [batch, frames, codebooks] or [batch, frames], got {tuple(idx.shape)}")

    def _embed_coarse(self, coarse_idx, past_length=0):
        bsz, frames = coarse_idx.shape
        if past_length + frames > self.config.block_size:
            raise ValueError(
                f"past length {past_length} with current length {frames} exceeds block_size {self.config.block_size}"
            )
        tok_emb = self.transformer.wte(coarse_idx)
        pos = torch.arange(past_length, past_length + frames, dtype=torch.long, device=coarse_idx.device)
        pos_emb = self.transformer.wpe(pos)
        return self.transformer.drop(tok_emb + pos_emb)

    def _run_backbone(self, coarse_idx, past_kvs=None, use_cache=False):
        if past_kvs is None:
            past_kvs = [None] * len(self.transformer.h)
            past_length = 0
        else:
            if len(past_kvs) != len(self.transformer.h):
                raise ValueError("past_kvs length must match transformer depth")
            first_past = next((item for item in past_kvs if item is not None), None)
            past_length = first_past[0].size(2) if first_past is not None else 0
        x = self._embed_coarse(coarse_idx, past_length=past_length)
        presents = []
        for block, past_kv in zip(self.transformer.h, past_kvs):
            x, present = block(x, past_kv=past_kv, use_cache=use_cache)
            presents.append(present)
        x = self.transformer.ln_f(x)
        return x, presents

    def predict_fine_logits(self, hidden, coarse_ids):
        if coarse_ids.dim() == 2:
            coarse_emb = self.transformer.wte(coarse_ids)
        else:
            raise ValueError(f"Expected coarse_ids shaped [batch, frames], got {tuple(coarse_ids.shape)}")
        cond = torch.cat((hidden, coarse_emb), dim=-1)
        base = torch.tanh(self.fine_proj(cond))

        logits = []
        for idx, head in enumerate(self.fine_heads):
            codebook_emb = self.fine_codebook_emb.weight[idx].view(1, 1, -1)
            logits.append(head(base + codebook_emb))
        return torch.stack(logits, dim=2)

    def forward(self, idx, targets=None):
        coarse_idx = self._coarse_input(idx)
        hidden, _ = self._run_backbone(coarse_idx, use_cache=False)
        coarse_logits = self.coarse_head(hidden)

        if targets is None:
            return coarse_logits[:, [-1], :], None

        if targets.dim() != 3 or targets.size(-1) != self.config.num_codebooks:
            raise ValueError(
                f"Expected targets shaped [batch, frames, {self.config.num_codebooks}], got {tuple(targets.shape)}"
            )
        fine_logits = self.predict_fine_logits(hidden, targets[:, :, 0])
        coarse_loss = F.cross_entropy(
            coarse_logits.reshape(-1, self.config.codebook_size),
            targets[:, :, 0].reshape(-1),
            ignore_index=-1,
        )
        fine_loss = F.cross_entropy(
            fine_logits.reshape(-1, self.config.codebook_size),
            targets[:, :, 1:].reshape(-1),
            ignore_index=-1,
        )
        return coarse_logits, coarse_loss + (self.config.fine_loss_weight * fine_loss)

    def forward_with_kv_cache(self, idx, past_kvs=None):
        coarse_idx = self._coarse_input(idx)
        hidden, presents = self._run_backbone(coarse_idx, past_kvs=past_kvs, use_cache=True)
        coarse_logits = self.coarse_head(hidden)
        return coarse_logits, presents, hidden

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
def generate_audio_coarse_fine_frames(
    model,
    prompt_frames: torch.Tensor,
    *,
    max_new_frames: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    frames = prompt_frames
    coarse_prompt = prompt_frames[:, :, 0]
    window = coarse_prompt[:, -min(coarse_prompt.size(1), model.config.block_size) :]
    coarse_logits, cached_kvs, hidden = model.forward_with_kv_cache(window)
    cached_window_length = window.size(1)

    for _ in range(max_new_frames):
        logits = coarse_logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        coarse_next = torch.multinomial(probs, num_samples=1)

        fine_logits = model.predict_fine_logits(hidden[:, -1:, :], coarse_next)
        next_codes = [coarse_next]
        for codebook_idx in range(fine_logits.size(2)):
            codebook_logits = fine_logits[:, 0, codebook_idx, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(codebook_logits, min(top_k, codebook_logits.size(-1)))
                codebook_logits = codebook_logits.masked_fill(codebook_logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(codebook_logits, dim=-1)
            next_codes.append(torch.multinomial(probs, num_samples=1))
        next_frame = torch.cat(next_codes, dim=1).unsqueeze(1)
        frames = torch.cat((frames, next_frame), dim=1)

        if cached_window_length < model.config.block_size:
            coarse_logits, cached_kvs, hidden = model.forward_with_kv_cache(coarse_next, past_kvs=cached_kvs)
            cached_window_length += 1
        else:
            coarse_window = frames[:, -model.config.block_size :, 0]
            coarse_logits, cached_kvs, hidden = model.forward_with_kv_cache(coarse_window)
            cached_window_length = coarse_window.size(1)

    return frames
