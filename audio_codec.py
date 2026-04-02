import os
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.nn import functional as F
import torchaudio


class EncodecUnavailableError(RuntimeError):
    pass


@dataclass
class AudioTokenBatch:
    codes: torch.Tensor
    flattened_tokens: torch.Tensor
    sample_rate: int
    channels: int
    codebook_size: int
    num_codebooks: int
    normalized_wav: Optional[torch.Tensor] = None


def _require_encodec():
    try:
        from encodec import EncodecModel
        from encodec.utils import convert_audio
    except ImportError as exc:
        raise EncodecUnavailableError(
            "The `encodec` package is required for audio tokenization. "
            "Install it with `python3 -m pip install -U encodec`."
        ) from exc
    return EncodecModel, convert_audio


def load_encodec_model(model_name: str = "encodec_24khz", bandwidth: float = 6.0, device: str = "cpu"):
    EncodecModel, _ = _require_encodec()

    if model_name == "encodec_24khz":
        model = EncodecModel.encodec_model_24khz()
    elif model_name == "encodec_48khz":
        raise NotImplementedError(
            "The first dataset format only supports encodec_24khz. "
            "The 48 kHz model emits per-frame scale values that are not preserved "
            "by the current flattening scheme."
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model.set_target_bandwidth(float(bandwidth))
    model.to(device)
    model.eval()
    return model


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    wav, sample_rate = torchaudio.load(path)
    return wav, sample_rate


def chunk_waveform(wav: torch.Tensor, chunk_num_samples: Optional[int]) -> Iterable[torch.Tensor]:
    if not chunk_num_samples or wav.size(-1) <= chunk_num_samples:
        yield wav
        return

    for start in range(0, wav.size(-1), chunk_num_samples):
        end = start + chunk_num_samples
        chunk = wav[:, start:end]
        if chunk.size(-1) == 0:
            continue
        yield chunk


def flatten_codes(codes: torch.Tensor, codebook_size: int) -> torch.Tensor:
    if codes.dim() != 2:
        raise ValueError(f"Expected codes shaped [num_codebooks, time], got {tuple(codes.shape)}")

    num_codebooks, time_steps = codes.shape
    offsets = torch.arange(num_codebooks, device=codes.device, dtype=codes.dtype).unsqueeze(1) * codebook_size
    flattened = (codes + offsets).transpose(0, 1).reshape(time_steps * num_codebooks)
    return flattened


def unflatten_codes(flat_tokens: torch.Tensor, num_codebooks: int, codebook_size: int) -> torch.Tensor:
    if flat_tokens.numel() % num_codebooks != 0:
        raise ValueError("Flattened token count must be divisible by num_codebooks")

    time_steps = flat_tokens.numel() // num_codebooks
    codes = flat_tokens.view(time_steps, num_codebooks).transpose(0, 1).contiguous()
    offsets = torch.arange(num_codebooks, device=codes.device, dtype=codes.dtype).unsqueeze(1) * codebook_size
    return codes - offsets


def flat_tokens_to_frame_codes(flat_tokens: torch.Tensor, num_codebooks: int, codebook_size: int) -> torch.Tensor:
    if flat_tokens.shape[-1] % num_codebooks != 0 and flat_tokens.dim() == 1:
        raise ValueError("Flattened token count must be divisible by num_codebooks")

    if flat_tokens.dim() == 1:
        frame_tokens = flat_tokens.view(-1, num_codebooks)
    elif flat_tokens.dim() == 2 and flat_tokens.size(-1) == num_codebooks:
        frame_tokens = flat_tokens
    elif flat_tokens.dim() == 3 and flat_tokens.size(-1) == num_codebooks:
        frame_tokens = flat_tokens
    else:
        raise ValueError(
            f"Expected flattened tokens shaped [tokens] or frame tokens [..., num_codebooks], got {tuple(flat_tokens.shape)}"
        )

    offsets = torch.arange(num_codebooks, device=frame_tokens.device, dtype=frame_tokens.dtype) * codebook_size
    return frame_tokens - offsets


def frame_codes_to_flat_tokens(frame_codes: torch.Tensor, codebook_size: int) -> torch.Tensor:
    if frame_codes.dim() != 2:
        raise ValueError(f"Expected frame codes shaped [time, num_codebooks], got {tuple(frame_codes.shape)}")
    num_codebooks = frame_codes.size(1)
    offsets = torch.arange(num_codebooks, device=frame_codes.device, dtype=frame_codes.dtype) * codebook_size
    return (frame_codes + offsets).reshape(-1)


def encode_waveform(
    model,
    wav: torch.Tensor,
    sample_rate: int,
    *,
    device: str = "cpu",
    codebook_size: int = 1024,
) -> AudioTokenBatch:
    _, convert_audio = _require_encodec()

    wav = convert_audio(wav, sample_rate, model.sample_rate, model.channels)
    normalized_wav = wav.cpu()
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        encoded_frames = model.encode(wav)

    codes = torch.cat([frame[0] for frame in encoded_frames], dim=-1).squeeze(0).cpu()
    flattened_tokens = flatten_codes(codes, codebook_size=codebook_size)

    return AudioTokenBatch(
        codes=codes,
        flattened_tokens=flattened_tokens,
        sample_rate=model.sample_rate,
        channels=model.channels,
        codebook_size=codebook_size,
        num_codebooks=codes.size(0),
        normalized_wav=normalized_wav,
    )


def decode_codes(model, codes: torch.Tensor, *, device: str = "cpu") -> torch.Tensor:
    if codes.dim() != 2:
        raise ValueError(f"Expected codes shaped [num_codebooks, time], got {tuple(codes.shape)}")

    encoded_frames = [(codes.unsqueeze(0).to(device), None)]
    with torch.no_grad():
        wav = model.decode(encoded_frames)
    return wav.squeeze(0).cpu()


def encode_audio_file(
    path: str,
    model,
    *,
    device: str = "cpu",
    codebook_size: int = 1024,
) -> AudioTokenBatch:
    wav, sample_rate = load_audio(path)
    return encode_waveform(
        model,
        wav,
        sample_rate,
        device=device,
        codebook_size=codebook_size,
    )


def save_waveform(path: str, wav: torch.Tensor, sample_rate: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchaudio.save(path, wav.cpu(), sample_rate)


@torch.no_grad()
def generate_audio_tokens(
    model,
    prompt_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    num_codebooks: int,
    codebook_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate flattened EnCodec tokens while constraining each step to the
    codebook-specific token range implied by the interleaving pattern.
    """
    idx = prompt_tokens
    max_new_tokens = max_new_tokens - (max_new_tokens % num_codebooks)

    cached_logits = None
    cached_kvs = None
    cached_window_length = 0
    if hasattr(model, "forward_with_kv_cache"):
        prompt_window = idx[:, -min(idx.size(1), model.config.block_size):]
        if prompt_window.size(1) > 0:
            cached_logits, cached_kvs = model.forward_with_kv_cache(prompt_window)
            cached_window_length = prompt_window.size(1)

    for _ in range(max_new_tokens):
        if cached_logits is not None:
            logits = cached_logits[:, -1, :] / temperature
        else:
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

        next_position = idx.size(1)
        codebook_index = next_position % num_codebooks
        allowed_start = codebook_index * codebook_size
        allowed_end = allowed_start + codebook_size

        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits[:, allowed_start:allowed_end] = logits[:, allowed_start:allowed_end]
        logits = masked_logits

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, codebook_size))
            logits[logits < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if cached_kvs is not None and cached_window_length < model.config.block_size:
            cached_logits, cached_kvs = model.forward_with_kv_cache(idx_next, past_kvs=cached_kvs)
            cached_window_length += 1
        else:
            cached_logits = None
            cached_kvs = None

    return idx
