"""
Sample V5 audio using:
1. semantic AR continuation
2. acoustic decoding from semantic tokens
3. EnCodec waveform reconstruction
"""

import os
import pickle
import warnings
from contextlib import nullcontext

import torch

from audio_acoustic_decoder import AudioAcousticDecoder
from audio_acoustic_decoder import AudioAcousticDecoderConfig
from audio_acoustic_decoder import decode_codec_frames_from_semantics
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

# Keep notebook / CLI sampling output readable by silencing known third-party
# deprecation and transformer implementation warnings we are not acting on here.
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
warnings.filterwarnings(
    "ignore",
    message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.(load_with_torchcodec|save_with_torchcodec)` under the hood\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torio\.io\._streaming_media_(decoder|encoder)\.StreamingMedia(Decoder|Encoder) has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\._backend\..*")

# -----------------------------------------------------------------------------
semantic_out_dir = 'out-audio-semantic-v5'
acoustic_out_dir = 'out-audio-acoustic-decoder-v5'
prompt_audio = ''
output_wav = 'out-audio-v5/generated.wav'
prompt_max_seconds = 0.0
max_new_seconds = 2.0
semantic_source = 'predicted' # 'predicted' or 'ground_truth'
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
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path}. This usually means the checkpoint was trained with an older "
            f"architecture and needs to be retrained from scratch for the current code.\nOriginal error: {exc}"
        ) from exc
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
acoustic_model, acoustic_checkpoint = load_model(AudioAcousticDecoder, AudioAcousticDecoderConfig, acoustic_out_dir)

semantic_dataset = semantic_checkpoint.get('config', {}).get('dataset', 'audio_semantic_codec')
acoustic_dataset = acoustic_checkpoint.get('config', {}).get('dataset', semantic_dataset)
if semantic_dataset != acoustic_dataset:
    raise ValueError(
        f"Semantic checkpoint dataset {semantic_dataset!r} does not match acoustic dataset {acoustic_dataset!r}"
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
    semantic_batch.normalized_wav if semantic_batch.normalized_wav is not None else prompt_wav,
    meta['sample_rate'],
    target_frames=codec_batch.codes.size(1),
    frame_rate_hz=float(meta['codec_frame_rate_hz_estimate']),
)

semantic_prompt_cpu = semantic_batch.tokens.to(torch.long)[None, :]
semantic_prompt = semantic_prompt_cpu.to(device)
codec_frame_rate = float(meta['codec_frame_rate_hz_estimate'])
semantic_rate = float(meta['semantic_rate_hz_estimate'])
max_new_semantic = max(1, int(round(max_new_seconds * semantic_rate)))
target_frames = codec_batch.codes.size(1) + int(round(max_new_seconds * codec_frame_rate))

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
                raise ValueError(
                    "ground_truth semantic_source requested, but the prompt clip has no remaining future semantic tokens. "
                    "Use a shorter prompt_max_seconds or a longer source file."
                )
            generated_semantic = torch.cat((semantic_prompt_cpu, future_semantic[None, :]), dim=1).to(device)
        else:
            raise ValueError(f"Unsupported semantic_source={semantic_source!r}")
        decoded_frames = decode_codec_frames_from_semantics(
            acoustic_model,
            generated_semantic,
            prompt_codec_frames=codec_batch.codes.transpose(0, 1).contiguous()[None, ...].to(device),
            prompt_prosody_features=prompt_prosody[None, ...].to(device),
            target_frames=int(round(max_new_seconds * codec_frame_rate)),
            temperature=temperature,
            top_k=top_k,
        )

continuation_codes = decoded_frames[0].transpose(0, 1).contiguous().cpu()
full_codes = torch.cat((codec_batch.codes, continuation_codes), dim=1)
generated_wav = decode_codes(codec, full_codes, device='cpu')
prompt_wav = codec_batch.normalized_wav if codec_batch.normalized_wav is not None else prompt_wav
continuation_wav = decode_codes(codec, continuation_codes, device='cpu')

os.makedirs(os.path.dirname(output_wav), exist_ok=True)
root, ext = os.path.splitext(output_wav)
prompt_path = f"{root}_prompt{ext}"
continuation_path = f"{root}_continuation{ext}"

save_waveform(prompt_path, prompt_wav, meta['sample_rate'])
save_waveform(output_wav, generated_wav, meta['sample_rate'])
save_waveform(continuation_path, continuation_wav, meta['sample_rate'])

print(f"Saved prompt audio to {prompt_path}")
print(f"Saved full audio to {output_wav}")
print(f"Saved continuation audio to {continuation_path}")
