import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import Block, LayerNorm


@dataclass
class AudioSemanticGPTConfig:
    block_size: int = 1024
    vocab_size: int = 1024
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class AudioSemanticGPT(nn.Module):
    def __init__(self, config: AudioSemanticGPTConfig):
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("vocab_size and block_size must be set")
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

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

    def forward(self, idx, targets=None):
        device = idx.device
        t = idx.size(1)
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size :]
            t = idx.size(1)
            if targets is not None:
                targets = targets[:, -self.config.block_size :]

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def forward_with_kv_cache(self, idx, past_kvs=None):
        device = idx.device
        t = idx.size(1)
        if t <= 0:
            raise ValueError("idx must contain at least one token")

        if past_kvs is None:
            past_kvs = [None] * len(self.transformer.h)
            past_length = 0
        else:
            if len(past_kvs) != len(self.transformer.h):
                raise ValueError("past_kvs length must match transformer depth")
            first_past = next((item for item in past_kvs if item is not None), None)
            past_length = first_past[0].size(2) if first_past is not None else 0

        if past_length + t > self.config.block_size:
            raise ValueError(
                f"past length {past_length} with current length {t} exceeds block_size {self.config.block_size}"
            )

        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        presents = []
        for block, past_kv in zip(self.transformer.h, past_kvs):
            x, present = block(x, past_kv=past_kv, use_cache=True)
            presents.append(present)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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
        layers, heads, q_width, steps = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * n_params + 12 * layers * heads * q_width * steps
        flops_per_fwdbwd = flops_per_token * steps
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised


@torch.no_grad()
def generate_semantic_tokens(
    model,
    prompt_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    idx = prompt_tokens

    cached_logits = None
    cached_kvs = None
    cached_window_length = 0
    if hasattr(model, "forward_with_kv_cache"):
        window = idx[:, -min(idx.size(1), model.config.block_size) :]
        if window.size(1) > 0:
            cached_logits, cached_kvs = model.forward_with_kv_cache(window)
            cached_window_length = window.size(1)

    for _ in range(max_new_tokens):
        if cached_logits is not None:
            logits = cached_logits[:, -1, :] / temperature
        else:
            idx_cond = idx[:, -model.config.block_size :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if cached_kvs is not None and cached_window_length < model.config.block_size:
            cached_logits, cached_kvs = model.forward_with_kv_cache(idx_next, past_kvs=cached_kvs)
            cached_window_length += 1
        else:
            window = idx[:, -min(idx.size(1), model.config.block_size) :]
            cached_logits, cached_kvs = model.forward_with_kv_cache(window)
            cached_window_length = window.size(1)

    return idx
