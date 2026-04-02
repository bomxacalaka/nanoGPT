"""
Sample V5.4 audio using:
1. semantic AR continuation
2. coarse decoder over codec codebook 0
3. fine decoder over residual codec codebooks 1..N-1
4. EnCodec waveform reconstruction
"""

import os
import pickle
import json
import warnings
from contextlib import nullcontext

import torch

from audio_critique_metrics import frame_codebook_accuracy
from audio_critique_metrics import normalized_edit_distance
from audio_critique_metrics import prosody_mae
from audio_critique_metrics import token_accuracy
from audio_codec import decode_codes
from audio_codec import encode_waveform
from audio_codec import load_audio
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_coarse_decoder import AudioCoarseDecoder
from audio_coarse_decoder import AudioCoarseDecoderConfig
from audio_coarse_decoder import decode_coarse_codes_from_semantics
from audio_fine_decoder import AudioFineDecoder
from audio_fine_decoder import AudioFineDecoderConfig
from audio_fine_decoder import decode_fine_codes_from_semantics_and_coarse
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
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated in favor of `torch\.nn\.utils\.parametrizations\.weight_norm`\.",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\._backend\..*")

# -----------------------------------------------------------------------------
semantic_out_dir = 'out-audio-semantic-v5'
coarse_out_dir = 'out-audio-coarse-decoder-v5'
fine_out_dir = 'out-audio-fine-decoder-v5'
prompt_audio = ''
output_wav = 'out-audio-v5-two-stage/generated.wav'
prompt_max_seconds = 0.0
max_new_seconds = 2.0
semantic_source = 'predicted'  # 'predicted' or 'ground_truth'
coarse_source = 'predicted'  # 'predicted' or 'ground_truth'
write_metrics_json = True
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
fine_model, fine_checkpoint = load_model(AudioFineDecoder, AudioFineDecoderConfig, fine_out_dir)

semantic_dataset = semantic_checkpoint.get('config', {}).get('dataset', 'audio_semantic_codec')
coarse_dataset = coarse_checkpoint.get('config', {}).get('dataset', semantic_dataset)
fine_dataset = fine_checkpoint.get('config', {}).get('dataset', semantic_dataset)
if len({semantic_dataset, coarse_dataset, fine_dataset}) != 1:
    raise ValueError(
        f"Dataset mismatch across checkpoints: semantic={semantic_dataset!r}, coarse={coarse_dataset!r}, fine={fine_dataset!r}"
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
full_codec_batch = encode_waveform(codec, full_wav, sample_rate, device='cpu', codebook_size=meta['codebook_size'])
prompt_codec_batch = encode_waveform(codec, prompt_wav, sample_rate, device='cpu', codebook_size=meta['codebook_size'])
prompt_prosody = extract_prosody_features(
    prompt_codec_batch.normalized_wav if prompt_codec_batch.normalized_wav is not None else prompt_wav,
    meta['sample_rate'],
    target_frames=prompt_codec_batch.codes.size(1),
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

        if coarse_source == 'predicted':
            continuation_coarse = decode_coarse_codes_from_semantics(
                coarse_model,
                generated_semantic,
                prompt_codec_frames=prompt_codec_batch.codes.transpose(0, 1).contiguous()[None, ...].to(device),
                prompt_prosody_features=prompt_prosody[None, ...].to(device),
                prompt_valid_lengths=torch.tensor([prompt_codec_batch.codes.size(1)], dtype=torch.long, device=device),
                target_frames=target_frames,
                temperature=temperature,
                top_k=top_k,
            )
        elif coarse_source == 'ground_truth':
            prompt_frame_count = prompt_codec_batch.codes.size(1)
            future_coarse = full_codec_batch.codes[0, prompt_frame_count:prompt_frame_count + target_frames]
            if future_coarse.numel() < target_frames:
                raise ValueError("No future coarse codec frames available. Use a shorter prompt or longer source file.")
            continuation_coarse = future_coarse.to(torch.long)[None, :].to(device)
        else:
            raise ValueError(f"Unsupported coarse_source={coarse_source!r}")

        continuation_residual = decode_fine_codes_from_semantics_and_coarse(
            fine_model,
            generated_semantic,
            continuation_coarse,
            prompt_codec_frames=prompt_codec_batch.codes.transpose(0, 1).contiguous()[None, ...].to(device),
            prompt_prosody_features=prompt_prosody[None, ...].to(device),
            prompt_valid_lengths=torch.tensor([prompt_codec_batch.codes.size(1)], dtype=torch.long, device=device),
            temperature=temperature,
            top_k=top_k,
        )[0].cpu()

continuation_coarse_cpu = continuation_coarse[0].cpu()
continuation_codes = torch.cat(
    (
        continuation_coarse_cpu.unsqueeze(0),
        continuation_residual.transpose(0, 1).contiguous(),
    ),
    dim=0,
)
full_codes = torch.cat((prompt_codec_batch.codes, continuation_codes), dim=1)
generated_wav = decode_codes(codec, full_codes, device='cpu')
continuation_wav = decode_codes(codec, continuation_codes, device='cpu')
prompt_wav = prompt_codec_batch.normalized_wav if prompt_codec_batch.normalized_wav is not None else prompt_wav

os.makedirs(os.path.dirname(output_wav), exist_ok=True)
root, ext = os.path.splitext(output_wav)
prompt_path = f"{root}_prompt{ext}"
continuation_path = f"{root}_continuation{ext}"
target_continuation_path = f"{root}_target{ext}"
metrics_path = f"{root}_metrics.json"

full_normalized_wav = full_codec_batch.normalized_wav if full_codec_batch.normalized_wav is not None else full_wav
prompt_normalized_wav = prompt_codec_batch.normalized_wav if prompt_codec_batch.normalized_wav is not None else prompt_wav
prompt_samples = int(prompt_normalized_wav.size(-1))
target_continuation_wav = full_normalized_wav[:, prompt_samples:prompt_samples + continuation_wav.size(-1)]
target_available = target_continuation_wav.size(-1) > 0

save_waveform(prompt_path, prompt_wav, meta['sample_rate'])
save_waveform(output_wav, generated_wav, meta['sample_rate'])
save_waveform(continuation_path, continuation_wav, meta['sample_rate'])

metrics = {
    "semantic_dataset": semantic_dataset,
    "semantic_source": semantic_source,
    "coarse_source": coarse_source,
    "generated_semantic_tokens": int(generated_semantic.size(1) - semantic_prompt.size(1)),
    "generated_codec_frames": int(continuation_codes.size(1)),
    "paths": {
        "prompt": prompt_path,
        "full": output_wav,
        "continuation": continuation_path,
        "target_continuation": target_continuation_path if target_available else None,
        "metrics": metrics_path,
        "source_audio": prompt_audio,
    },
    "critique": {},
}

prompt_semantic_len = int(semantic_batch.tokens.numel())
generated_semantic_continuation = generated_semantic[0, prompt_semantic_len:].detach().cpu()
target_semantic_continuation = semantic_batch_full.tokens[prompt_semantic_len:prompt_semantic_len + generated_semantic_continuation.numel()].detach().cpu()
reencoded_generated_semantic = semantic_tokenizer.encode_waveform(continuation_wav, meta['sample_rate']).tokens.detach().cpu()
metrics["critique"]["semantic"] = {
    "generated_vs_target": token_accuracy(generated_semantic_continuation, target_semantic_continuation),
    "reencoded_audio_vs_target": token_accuracy(reencoded_generated_semantic, target_semantic_continuation),
    "reencoded_audio_vs_target_norm_edit_distance": normalized_edit_distance(
        reencoded_generated_semantic,
        target_semantic_continuation,
    ),
}

generated_codec_batch = encode_waveform(
    codec,
    continuation_wav,
    meta['sample_rate'],
    device='cpu',
    codebook_size=meta['codebook_size'],
)
target_codec_frames = full_codec_batch.codes[:, prompt_codec_batch.codes.size(1):prompt_codec_batch.codes.size(1) + continuation_codes.size(1)]
metrics["critique"]["codec"] = {
    "all_codebooks": frame_codebook_accuracy(generated_codec_batch.codes, target_codec_frames),
    "coarse_codebook_0": token_accuracy(generated_codec_batch.codes[0], target_codec_frames[0]),
    "residual_codebooks_1plus": frame_codebook_accuracy(
        generated_codec_batch.codes[1:],
        target_codec_frames[1:],
    ) if generated_codec_batch.codes.size(0) > 1 and target_codec_frames.size(0) > 1 else None,
}

if target_available:
    save_waveform(target_continuation_path, target_continuation_wav, meta['sample_rate'])
    compare_frames = min(generated_codec_batch.codes.size(1), target_codec_frames.size(1))
    generated_prosody = extract_prosody_features(
        continuation_wav,
        meta['sample_rate'],
        target_frames=max(1, compare_frames),
        frame_rate_hz=float(meta['codec_frame_rate_hz_estimate']),
    )
    target_prosody = extract_prosody_features(
        target_continuation_wav,
        meta['sample_rate'],
        target_frames=max(1, compare_frames),
        frame_rate_hz=float(meta['codec_frame_rate_hz_estimate']),
    )
    metrics["critique"]["prosody"] = prosody_mae(
        generated_prosody,
        target_prosody,
        meta.get("prosody_feature_names", ["log_pitch_hz", "log_energy", "voiced"]),
    )
else:
    metrics["critique"]["prosody"] = {
        "compare_frames": 0,
        "mean_abs_error": None,
        "per_feature_mae": {},
    }

if write_metrics_json:
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

print(f"Saved prompt audio to {prompt_path}")
print(f"Saved two-stage full audio to {output_wav}")
print(f"Saved two-stage continuation audio to {continuation_path}")
if target_available:
    print(f"Saved target continuation audio to {target_continuation_path}")
if write_metrics_json:
    print(f"Saved critique metrics to {metrics_path}")
print(json.dumps(metrics, indent=2, sort_keys=True))
