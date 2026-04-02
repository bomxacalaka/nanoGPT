"""
Sample coarse-only V5.3 audio using:
1. semantic AR continuation
2. coarse decoder over the first codec codebook
3. rough waveform reconstruction by filling residual codebooks from the last prompt frame

This is intentionally a diagnostic path for intelligibility, not a final high-fidelity decoder.
"""

import os
import pickle
import warnings
from contextlib import nullcontext

import torch

from audio_coarse_decoder import AudioCoarseDecoder
from audio_coarse_decoder import AudioCoarseDecoderConfig
from audio_coarse_decoder import decode_coarse_codes_from_semantics
from audio_codec import decode_codes
from audio_codec import encode_waveform
from audio_codec import load_audio
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_prosody import extract_prosody_features
from audio_semantic_model import AudioSemanticGPT
from audio_semantic_model import AudioSemanticGPTConfig
from audio_semantic_model import generate_semantic_tokens
from audio_semantic_tokenizer import SemanticTokenizer

warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\._backend\..*")
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`\.",
    category=FutureWarning,
)

# -----------------------------------------------------------------------------
semantic_out_dir = 'out-audio-semantic-v5'
coarse_out_dir = 'out-audio-coarse-decoder-v5'
prompt_audio = ''
output_wav = 'out-audio-v5-coarse/generated.wav'
prompt_max_seconds = 0.0
max_new_seconds = 2.0
semantic_source = 'predicted'  # 'predicted' or 'ground_truth'
residual_fill = 'repeat_last'  # 'repeat_last' or 'zeros'
temperature = 0.9
top_k = 100
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------


def maybe_trim_waveform(wav, sample_rate, max_seconds):
    if max_seconds <= 0:
        return wav
    max_samples = int(sample_rate * max_seconds)
    if wav.size(-1) <= max_samples:
        return wav
    return wav[..., :max_samples]


def load_model(model_cls, config_cls, out_dir):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = model_cls(config_cls(**checkpoint['model_args']))
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model, checkpoint


def build_full_codes(prompt_codes: torch.Tensor, coarse_continuation: torch.Tensor, *, fill_mode: str) -> torch.Tensor:
    num_codebooks = prompt_codes.size(0)
    if coarse_continuation.dim() != 1:
        raise ValueError(
            f"Expected coarse_continuation shaped [frames], got {tuple(coarse_continuation.shape)}"
        )
    continuation_frames = coarse_continuation.size(0)
    if fill_mode == 'repeat_last':
        fill_frame = prompt_codes[:, -1:].expand(num_codebooks, continuation_frames).clone()
    elif fill_mode == 'zeros':
        fill_frame = torch.zeros((num_codebooks, continuation_frames), dtype=prompt_codes.dtype)
    else:
        raise ValueError(f"Unsupported residual_fill={fill_mode!r}")
    fill_frame[0, :] = coarse_continuation
    continuation_codes = fill_frame
    full_codes = torch.cat((prompt_codes, continuation_codes), dim=1)
    return full_codes, continuation_codes


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if not prompt_audio:
    raise ValueError('Set --prompt_audio=/path/to/file.wav')

semantic_model, semantic_checkpoint = load_model(AudioSemanticGPT, AudioSemanticGPTConfig, semantic_out_dir)
coarse_model, coarse_checkpoint = load_model(AudioCoarseDecoder, AudioCoarseDecoderConfig, coarse_out_dir)

semantic_dataset = semantic_checkpoint.get('config', {}).get('dataset', 'audio_semantic_codec')
coarse_dataset = coarse_checkpoint.get('config', {}).get('dataset', semantic_dataset)
if semantic_dataset != coarse_dataset:
    raise ValueError(
        f"Semantic checkpoint dataset {semantic_dataset!r} does not match coarse dataset {coarse_dataset!r}"
    )
meta_path = os.path.join('data', semantic_dataset, 'meta.pkl')
with open(meta_path, 'rb') as handle:
    meta = pickle.load(handle)

semantic_tokenizer = SemanticTokenizer.from_metadata(meta['semantic_tokenizer'])
codec = load_encodec_model(
    model_name=meta['codec_model_name'],
    bandwidth=meta['bandwidth'],
    device='cpu',
)

full_wav, sample_rate = load_audio(prompt_audio)
prompt_wav = maybe_trim_waveform(full_wav, sample_rate, prompt_max_seconds)
semantic_batch_full = semantic_tokenizer.encode_waveform(full_wav, sample_rate)
semantic_batch = semantic_tokenizer.encode_waveform(prompt_wav, sample_rate)
codec_batch = encode_waveform(codec, prompt_wav, sample_rate, device='cpu', codebook_size=meta['codebook_size'])
prompt_prosody = extract_prosody_features(
    codec_batch.normalized_wav if codec_batch.normalized_wav is not None else prompt_wav,
    meta['sample_rate'],
    target_frames=codec_batch.codes.size(1),
    frame_rate_hz=float(meta['codec_frame_rate_hz_estimate']),
)

semantic_prompt_cpu = semantic_batch.tokens.to(torch.long)[None, :]
semantic_prompt = semantic_prompt_cpu.to(device)
codec_frame_rate = float(meta['codec_frame_rate_hz_estimate'])
semantic_rate = float(meta['semantic_rate_hz_estimate'])
max_new_semantic = max(1, int(round(max_new_seconds * semantic_rate)))
target_frames = int(round(max_new_seconds * codec_frame_rate))

with torch.no_grad():
    with ctx:
        if semantic_source == 'predicted':
            generated_semantic = generate_semantic_tokens(
                semantic_model,
                semantic_prompt,
                max_new_tokens=max_new_semantic,
                temperature=temperature,
                top_k=top_k,
            )
        elif semantic_source == 'ground_truth':
            full_semantic = semantic_batch_full.tokens.to(torch.long)
            prompt_semantic_len = semantic_batch.tokens.numel()
            future_semantic = full_semantic[prompt_semantic_len:prompt_semantic_len + max_new_semantic]
            if future_semantic.numel() == 0:
                raise ValueError("No future semantic tokens available. Use a shorter prompt or longer source file.")
            generated_semantic = torch.cat((semantic_prompt_cpu, future_semantic[None, :]), dim=1).to(device)
        else:
            raise ValueError(f"Unsupported semantic_source={semantic_source!r}")

        coarse_continuation = decode_coarse_codes_from_semantics(
            coarse_model,
            generated_semantic,
            prompt_codec_frames=codec_batch.codes.transpose(0, 1).contiguous()[None, ...].to(device),
            prompt_prosody_features=prompt_prosody[None, ...].to(device),
            prompt_valid_lengths=torch.tensor([codec_batch.codes.size(1)], dtype=torch.long, device=device),
            target_frames=target_frames,
            temperature=temperature,
            top_k=top_k,
        )[0].cpu()

full_codes, continuation_codes = build_full_codes(
    codec_batch.codes,
    coarse_continuation,
    fill_mode=residual_fill,
)
generated_wav = decode_codes(codec, full_codes, device='cpu')
continuation_wav = decode_codes(codec, continuation_codes, device='cpu')
prompt_wav = codec_batch.normalized_wav if codec_batch.normalized_wav is not None else prompt_wav

os.makedirs(os.path.dirname(output_wav), exist_ok=True)
root, ext = os.path.splitext(output_wav)
prompt_path = f"{root}_prompt{ext}"
continuation_path = f"{root}_continuation{ext}"

save_waveform(prompt_path, prompt_wav, meta['sample_rate'])
save_waveform(output_wav, generated_wav, meta['sample_rate'])
save_waveform(continuation_path, continuation_wav, meta['sample_rate'])

print(f"Saved prompt audio to {prompt_path}")
print(f"Saved coarse full audio to {output_wav}")
print(f"Saved coarse continuation audio to {continuation_path}")
