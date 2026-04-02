from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import torch
import torchaudio

from audio_critic import AudioCritic
from audio_critic import AudioCriticConfig

# -----------------------------------------------------------------------------
manifest_path = "data/audio_semantic_codec/critic_manifest.jsonl"
out_dir = "out-audio-critic-v5"
sample_rate = 24000
batch_size = 8
max_steps = 10000
eval_interval = 200
log_interval = 20
learning_rate = 2e-4
weight_decay = 1e-2
grad_clip = 1.0
device = "cuda"
dtype = "float32"
compile = False

n_mels = 80
n_fft = 1024
hop_length = 256
win_length = 1024
max_frames = 512
base_channels = 64
hidden_dim = 256
dropout = 0.1
# -----------------------------------------------------------------------------
exec(open("configurator.py").read())


def load_manifest(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in manifest {path}")
    return rows


def split_rows(rows, val_frac=0.1, seed=1337):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_count = max(1, int(math.ceil(len(rows) * val_frac))) if len(rows) > 1 else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        train_rows, val_rows = rows, []
    return train_rows, val_rows


def load_audio_file(path: str, target_sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sr, target_sample_rate)
    return wav


def collate_batch(rows, target_sample_rate):
    prompt = []
    generated = []
    target = []
    targets = {
        "realism": [],
        "intelligibility": [],
        "style_match": [],
        "prosody_match": [],
        "semantic_match": [],
    }
    for row in rows:
        prompt.append(load_audio_file(row["prompt_path"], target_sample_rate))
        generated.append(load_audio_file(row["generated_path"], target_sample_rate))
        target.append(load_audio_file(row["target_path"], target_sample_rate))
        for key in targets:
            targets[key].append(float(row["targets"][key]))

    max_prompt = max(w.size(-1) for w in prompt)
    max_generated = max(w.size(-1) for w in generated)
    max_target = max(w.size(-1) for w in target)

    def pad_stack(wavs, max_len):
        padded = []
        for wav in wavs:
            if wav.size(-1) < max_len:
                wav = torch.nn.functional.pad(wav, (0, max_len - wav.size(-1)))
            padded.append(wav)
        return torch.stack(padded)

    batch = {
        "prompt": pad_stack(prompt, max_prompt),
        "generated": pad_stack(generated, max_generated),
        "target": pad_stack(target, max_target),
        "targets": {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()},
    }
    return batch


def compute_loss(pred: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
    losses = {}
    total = 0.0
    for key, target in targets.items():
        loss = torch.nn.functional.mse_loss(pred[key], target)
        losses[key] = loss
        total = total + loss
    return total, losses


def evaluate(model, rows, target_sample_rate, batch_size, device):
    if not rows:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start + batch_size]
            batch = collate_batch(batch_rows, target_sample_rate)
            prompt = batch["prompt"].to(device)
            generated = batch["generated"].to(device)
            target = batch["target"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            pred = model(prompt, generated, target)
            loss, _ = compute_loss(pred, targets)
            total += float(loss.item()) * len(batch_rows)
            count += len(batch_rows)
    return total / max(1, count)


def main():
    os.makedirs(out_dir, exist_ok=True)
    rows = load_manifest(manifest_path)
    train_rows, val_rows = split_rows(rows)
    print(f"Loaded {len(rows)} critic examples | train={len(train_rows)} val={len(val_rows)}")

    config = AudioCriticConfig(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        max_frames=max_frames,
        base_channels=base_channels,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model = AudioCritic(config).to(device)
    if compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val = float("inf")
    rng = random.Random(1337)
    for step in range(max_steps + 1):
        batch_rows = [train_rows[rng.randrange(len(train_rows))] for _ in range(batch_size)]
        batch = collate_batch(batch_rows, sample_rate)
        prompt = batch["prompt"].to(device)
        generated = batch["generated"].to(device)
        target = batch["target"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(prompt, generated, target)
        loss, losses = compute_loss(pred, targets)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % log_interval == 0:
            print(
                f"step {step}: loss {loss.item():.4f} "
                + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()),
                flush=True,
            )
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_rows, sample_rate, batch_size, device) if val_rows else None
            if val_loss is not None:
                print(f"eval step {step}: val_loss={val_loss:.4f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = {
                        "model": model.state_dict(),
                        "model_args": asdict(config),
                        "config": {
                            "manifest_path": manifest_path,
                            "out_dir": out_dir,
                        },
                        "step": step,
                        "best_val_loss": best_val,
                    }
                    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
                    print(f"saved checkpoint to {os.path.join(out_dir, 'ckpt.pt')}", flush=True)


if __name__ == "__main__":
    main()
