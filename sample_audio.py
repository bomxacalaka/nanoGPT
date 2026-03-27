"""
Sample audio continuation from a trained model over flattened EnCodec tokens.
"""

import os
import pickle
from contextlib import nullcontext

import torch
import torchaudio

from audio_codec import decode_codes
from audio_codec import encode_audio_file
from audio_codec import generate_audio_tokens
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_codec import unflatten_codes
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out-audio-codec'
prompt_audio = ''
output_wav = 'out-audio-codec/generated.wav'
prompt_max_seconds = 0.0
max_new_seconds = 1.0
temperature = 0.9
top_k = 100
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


def load_checkpoint_model():
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
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


def maybe_trim_audio(path, max_seconds):
    if max_seconds <= 0:
        return path

    wav, sample_rate = torchaudio.load(path)
    max_samples = int(max_seconds * sample_rate)
    if wav.size(-1) <= max_samples:
        return path

    root, ext = os.path.splitext(path)
    trimmed_path = f"{root}_trimmed{ext}"
    torchaudio.save(trimmed_path, wav[..., :max_samples], sample_rate)
    return trimmed_path


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

prompt_audio = maybe_trim_audio(prompt_audio, prompt_max_seconds)

model, checkpoint = load_checkpoint_model()
dataset = checkpoint.get('config', {}).get('dataset', 'audio_codec')
meta_path = os.path.join('data', dataset, 'meta.pkl')
with open(meta_path, 'rb') as handle:
    meta = pickle.load(handle)

codec = load_encodec_model(
    model_name=meta['model_name'],
    bandwidth=meta['bandwidth'],
    device='cpu',
)
prompt_batch = encode_audio_file(
    prompt_audio,
    codec,
    device='cpu',
    codebook_size=meta['codebook_size'],
)

tokens_per_second = meta['tokens_per_second']
max_new_tokens = int(max_new_seconds * tokens_per_second)
num_codebooks = meta['num_codebooks']
codebook_size = meta['codebook_size']

x = torch.tensor(prompt_batch.flattened_tokens.tolist(), dtype=torch.long, device=device)[None, ...]
if x.size(1) > model.config.block_size:
    prompt_seconds = x.size(1) / tokens_per_second
    visible_seconds = model.config.block_size / tokens_per_second
    print(
        f"warning: prompt is {prompt_seconds:.2f}s but block_size only covers "
        f"{visible_seconds:.2f}s; generation is conditioned on the tail only"
    )

with torch.no_grad():
    with ctx:
        y = generate_audio_tokens(
            model,
            x,
            max_new_tokens=max_new_tokens,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            temperature=temperature,
            top_k=top_k,
        )

all_tokens = y[0].tolist()
valid_token_count = len(all_tokens) - (len(all_tokens) % num_codebooks)
all_tokens = all_tokens[:valid_token_count]

codes = unflatten_codes(
    torch.tensor(all_tokens, dtype=torch.long),
    num_codebooks=num_codebooks,
    codebook_size=codebook_size,
)
wav = decode_codes(codec, codes, device='cpu')
save_waveform(output_wav, wav, meta['sample_rate'])

print(f"Saved generated audio to {output_wav}")
