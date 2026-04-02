"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
import sys
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from audio_frame_model import AudioFrameGPT
from audio_frame_model import AudioFrameGPTConfig
from audio_frame_model import generate_audio_frames
from audio_coarse_fine_model import AudioCoarseFineGPT
from audio_coarse_fine_model import AudioCoarseFineGPTConfig
from audio_coarse_fine_model import generate_audio_coarse_fine_frames
from audio_semantic_model import AudioSemanticGPT
from audio_semantic_model import AudioSemanticGPTConfig
from audio_coarse_decoder import AudioCoarseDecoder
from audio_coarse_decoder import AudioCoarseDecoderConfig
from audio_fine_decoder import AudioFineDecoder
from audio_fine_decoder import AudioFineDecoderConfig
from audio_acoustic_decoder import AudioAcousticDecoder
from audio_acoustic_decoder import AudioAcousticDecoderConfig
from tokenizer import Tokenizer
from colour_print import cprint
from audio_codec import decode_codes
from audio_codec import encode_audio_file
from audio_codec import encode_waveform
from audio_codec import frame_codes_to_flat_tokens
from audio_codec import flat_tokens_to_frame_codes
from audio_codec import generate_audio_tokens
from audio_codec import load_encodec_model
from audio_codec import save_waveform
from audio_codec import unflatten_codes

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
eval_interval_extra = 6000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
checkpoint_interval = 0 # if > 0, save an additional step-tagged checkpoint every N iters
checkpoint_keep_last = 0 # if > 0, keep only the most recent N step-tagged checkpoints
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
resume_ckpt_path = ''
reset_optimizer_on_resume = False
reset_iteration_on_resume = False
reset_best_val_loss_on_resume = False
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_model_upload = False # upload the model checkpoint to wandb
# tensorboard logging
tensorboard_log = False
tensorboard_run_name = ''
tensorboard_flush_secs = 10
console_log_file = 'train.log'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
model_type = 'gpt' # 'gpt', 'audio_frame_gpt', 'audio_coarse_fine_gpt', 'audio_semantic_gpt', 'audio_coarse_decoder', 'audio_fine_decoder', or 'audio_acoustic_decoder'
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# audio sampling
sample_prompt_audio = ''
sample_prompt_seconds = 0.0
sample_max_new_seconds = 0.5
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = 'samples'
style_prompt_frames = 96
style_prompt_dropout_prob = 0.5
style_prompt_zero_prob = 0.1
style_prompt_min_frames = 8
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)() for stream in self.streams)

    def fileno(self):
        return self.streams[0].fileno()

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    if console_log_file:
        log_path = os.path.join(out_dir, console_log_file)
        log_fp = open(log_path, 'a', buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_fp)
        sys.stderr = TeeStream(sys.stderr, log_fp)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
data_bin_dtype = np.uint16
semantic_bin_dtype = np.uint16
codec_bin_dtype = np.uint16
AUDIO_FRAME_DATA_CACHE = {}
V5_DATA_CACHE = {}


def load_audio_frame_data(split):
    if split in AUDIO_FRAME_DATA_CACHE:
        return AUDIO_FRAME_DATA_CACHE[split]
    path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(path, dtype=data_bin_dtype, mode='r')
    separator_token_id = meta.get('separator_token_id')
    if separator_token_id is None:
        filtered = np.asarray(data, dtype=np.int64)
    else:
        filtered = np.asarray(data[data != separator_token_id], dtype=np.int64)
    AUDIO_FRAME_DATA_CACHE[split] = filtered
    return filtered


def load_v5_data(split):
    if split in V5_DATA_CACHE:
        return V5_DATA_CACHE[split]
    prefix = 'train' if split == 'train' else 'val'
    semantic = np.memmap(os.path.join(data_dir, f'{prefix}.semantic.bin'), dtype=semantic_bin_dtype, mode='r')
    codec = np.memmap(os.path.join(data_dir, f'{prefix}.codec.bin'), dtype=codec_bin_dtype, mode='r')
    prosody = np.load(os.path.join(data_dir, f'{prefix}.prosody.npy'))
    align = np.load(os.path.join(data_dir, f'{prefix}.align.npy'))
    V5_DATA_CACHE[split] = {
        'semantic': semantic,
        'codec': codec,
        'prosody': prosody,
        'align': align,
    }
    return V5_DATA_CACHE[split]


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if dataset_type == 'audio_semantic_codec_v5':
        data = load_v5_data(split)
        semantic = data['semantic']
        codec = data['codec']
        prosody = data['prosody']
        align = data['align']
        if model_type == 'audio_semantic_gpt':
            semantic_window = block_size + 1
            valid_rows = align[align[:, 1] >= semantic_window]
            if len(valid_rows) == 0:
                raise ValueError(
                    f"No utterances in split {split} are long enough for semantic_window={semantic_window}"
                )
            chosen = np.atleast_2d(valid_rows[np.random.randint(len(valid_rows), size=batch_size)]).tolist()
            semantic_rows = []
            target_rows = []
            for semantic_offset, semantic_length, _, _ in chosen:
                max_start = int(semantic_length - semantic_window)
                start = 0 if max_start <= 0 else int(torch.randint(max_start + 1, (1,)).item())
                semantic_start = int(semantic_offset) + start
                sequence = torch.from_numpy((semantic[semantic_start:semantic_start+semantic_window]).astype(np.int64))
                semantic_rows.append(sequence[:block_size])
                target_rows.append(sequence[1:])
            x = torch.stack(semantic_rows)
            y = torch.stack(target_rows)
        elif model_type == 'audio_coarse_decoder':
            ratio = int(meta['semantic_to_codec_ratio'])
            num_codebooks = int(meta['num_codebooks'])
            semantic_window = max(1, math.ceil(block_size / ratio))
            frame_window = block_size
            valid_rows = align[(align[:, 1] >= semantic_window) & (align[:, 3] >= frame_window)]
            if len(valid_rows) == 0:
                raise ValueError(
                    f"No utterances in split {split} are long enough for semantic_window={semantic_window} "
                    f"and frame_window={frame_window}"
                )
            chosen = np.atleast_2d(valid_rows[np.random.randint(len(valid_rows), size=batch_size)]).tolist()
            semantic_rows = []
            coarse_rows = []
            prompt_style_rows = []
            prompt_prosody_rows = []
            prompt_length_rows = []
            for semantic_offset, semantic_length, codec_frame_offset, codec_frame_length in chosen:
                max_start = int(min(semantic_length - semantic_window, codec_frame_length - frame_window))
                start = 0 if max_start <= 0 else int(torch.randint(max_start + 1, (1,)).item())
                semantic_start = int(semantic_offset) + start
                codec_start_frame = int(codec_frame_offset) + start
                codec_start_token = codec_start_frame * num_codebooks
                codec_span_tokens = frame_window * num_codebooks
                prompt_frame_count = max(1, min(int(style_prompt_frames), int(codec_frame_length)))
                prompt_start_token = int(codec_frame_offset) * num_codebooks
                prompt_span_tokens = prompt_frame_count * num_codebooks
                prompt_prosody_slice = torch.from_numpy(
                    prosody[int(codec_frame_offset):int(codec_frame_offset) + prompt_frame_count].astype(np.float32)
                )
                semantic_rows.append(torch.from_numpy((semantic[semantic_start:semantic_start+semantic_window]).astype(np.int64)))
                codec_slice = torch.from_numpy((codec[codec_start_token:codec_start_token+codec_span_tokens]).astype(np.int64))
                prompt_slice = torch.from_numpy((codec[prompt_start_token:prompt_start_token+prompt_span_tokens]).astype(np.int64))
                coarse_rows.append(
                    flat_tokens_to_frame_codes(
                        codec_slice.view(frame_window, num_codebooks),
                        num_codebooks=num_codebooks,
                        codebook_size=int(meta['codec_vocab_size']),
                    )[:, 0]
                )
                prompt_codes = flat_tokens_to_frame_codes(
                    prompt_slice.view(prompt_frame_count, num_codebooks),
                    num_codebooks=num_codebooks,
                    codebook_size=int(meta['codec_vocab_size']),
                )
                keep_prompt_frames = prompt_frame_count
                if split == 'train' and style_prompt_dropout_prob > 0:
                    if torch.rand(1).item() < style_prompt_dropout_prob:
                        if torch.rand(1).item() < style_prompt_zero_prob:
                            keep_prompt_frames = 0
                        else:
                            min_keep = min(prompt_frame_count, max(1, int(style_prompt_min_frames)))
                            if min_keep < prompt_frame_count:
                                keep_prompt_frames = int(torch.randint(min_keep, prompt_frame_count + 1, (1,)).item())
                if keep_prompt_frames < prompt_frame_count:
                    prompt_codes = prompt_codes.clone()
                    prompt_prosody_slice = prompt_prosody_slice.clone()
                    prompt_codes[keep_prompt_frames:, :] = 0
                    prompt_prosody_slice[keep_prompt_frames:, :] = 0
                prompt_style_rows.append(prompt_codes)
                prompt_prosody_rows.append(prompt_prosody_slice)
                prompt_length_rows.append(torch.tensor(keep_prompt_frames, dtype=torch.long))
            x = (
                torch.stack(semantic_rows),
                torch.stack(prompt_style_rows),
                torch.stack(prompt_prosody_rows),
                torch.stack(prompt_length_rows),
            )
            y = torch.stack(coarse_rows)
        elif model_type == 'audio_acoustic_decoder':
            ratio = int(meta['semantic_to_codec_ratio'])
            num_codebooks = int(meta['num_codebooks'])
            semantic_window = max(1, block_size // ratio)
            frame_window = semantic_window * ratio
            valid_rows = align[(align[:, 1] >= semantic_window) & (align[:, 3] >= frame_window)]
            if len(valid_rows) == 0:
                raise ValueError(
                    f"No utterances in split {split} are long enough for semantic_window={semantic_window} "
                    f"and frame_window={frame_window}"
                )
            chosen = np.atleast_2d(valid_rows[np.random.randint(len(valid_rows), size=batch_size)]).tolist()
            semantic_rows = []
            codec_rows = []
            prompt_style_rows = []
            prompt_prosody_rows = []
            prompt_length_rows = []
            for semantic_offset, semantic_length, codec_frame_offset, codec_frame_length in chosen:
                max_start = int(min(semantic_length - semantic_window, (codec_frame_length // ratio) - semantic_window))
                start = 0 if max_start <= 0 else int(torch.randint(max_start + 1, (1,)).item())
                semantic_start = int(semantic_offset) + start
                codec_start_frame = int(codec_frame_offset) + (start * ratio)
                codec_start_token = codec_start_frame * num_codebooks
                codec_span_tokens = frame_window * num_codebooks
                prompt_frame_count = max(1, min(int(style_prompt_frames), int(codec_frame_length)))
                prompt_start_token = int(codec_frame_offset) * num_codebooks
                prompt_span_tokens = prompt_frame_count * num_codebooks
                prompt_prosody_slice = torch.from_numpy(
                    prosody[int(codec_frame_offset):int(codec_frame_offset) + prompt_frame_count].astype(np.float32)
                )
                semantic_rows.append(torch.from_numpy((semantic[semantic_start:semantic_start+semantic_window]).astype(np.int64)))
                codec_slice = torch.from_numpy((codec[codec_start_token:codec_start_token+codec_span_tokens]).astype(np.int64))
                prompt_slice = torch.from_numpy((codec[prompt_start_token:prompt_start_token+prompt_span_tokens]).astype(np.int64))
                codec_rows.append(
                    flat_tokens_to_frame_codes(
                        codec_slice.view(frame_window, num_codebooks),
                        num_codebooks=num_codebooks,
                        codebook_size=int(meta['codec_vocab_size']),
                    )
                )
                prompt_codes = flat_tokens_to_frame_codes(
                    prompt_slice.view(prompt_frame_count, num_codebooks),
                    num_codebooks=num_codebooks,
                    codebook_size=int(meta['codec_vocab_size']),
                )
                keep_prompt_frames = prompt_frame_count
                if split == 'train' and style_prompt_dropout_prob > 0:
                    if torch.rand(1).item() < style_prompt_dropout_prob:
                        if torch.rand(1).item() < style_prompt_zero_prob:
                            keep_prompt_frames = 0
                        else:
                            min_keep = min(prompt_frame_count, max(1, int(style_prompt_min_frames)))
                            if min_keep < prompt_frame_count:
                                keep_prompt_frames = int(torch.randint(min_keep, prompt_frame_count + 1, (1,)).item())
                if keep_prompt_frames < prompt_frame_count:
                    prompt_codes = prompt_codes.clone()
                    prompt_prosody_slice = prompt_prosody_slice.clone()
                    prompt_codes[keep_prompt_frames:, :] = 0
                    prompt_prosody_slice[keep_prompt_frames:, :] = 0
                prompt_style_rows.append(prompt_codes)
                prompt_prosody_rows.append(prompt_prosody_slice)
                prompt_length_rows.append(torch.tensor(keep_prompt_frames, dtype=torch.long))
            x = (
                torch.stack(semantic_rows),
                torch.stack(prompt_style_rows),
                torch.stack(prompt_prosody_rows),
                torch.stack(prompt_length_rows),
            )
            y = torch.stack(codec_rows)
        elif model_type == 'audio_fine_decoder':
            ratio = int(meta['semantic_to_codec_ratio'])
            num_codebooks = int(meta['num_codebooks'])
            semantic_window = max(1, math.ceil(block_size / ratio))
            frame_window = block_size
            valid_rows = align[(align[:, 1] >= semantic_window) & (align[:, 3] >= frame_window)]
            if len(valid_rows) == 0:
                raise ValueError(
                    f"No utterances in split {split} are long enough for semantic_window={semantic_window} "
                    f"and frame_window={frame_window}"
                )
            chosen = np.atleast_2d(valid_rows[np.random.randint(len(valid_rows), size=batch_size)]).tolist()
            semantic_rows = []
            coarse_rows = []
            residual_rows = []
            prompt_style_rows = []
            prompt_prosody_rows = []
            prompt_length_rows = []
            for semantic_offset, semantic_length, codec_frame_offset, codec_frame_length in chosen:
                max_start = int(min(semantic_length - semantic_window, codec_frame_length - frame_window))
                start = 0 if max_start <= 0 else int(torch.randint(max_start + 1, (1,)).item())
                semantic_start = int(semantic_offset) + start
                codec_start_frame = int(codec_frame_offset) + start
                codec_start_token = codec_start_frame * num_codebooks
                codec_span_tokens = frame_window * num_codebooks
                prompt_frame_count = max(1, min(int(style_prompt_frames), int(codec_frame_length)))
                prompt_start_token = int(codec_frame_offset) * num_codebooks
                prompt_span_tokens = prompt_frame_count * num_codebooks
                prompt_prosody_slice = torch.from_numpy(
                    prosody[int(codec_frame_offset):int(codec_frame_offset) + prompt_frame_count].astype(np.float32)
                )
                semantic_rows.append(torch.from_numpy((semantic[semantic_start:semantic_start+semantic_window]).astype(np.int64)))
                codec_slice = torch.from_numpy((codec[codec_start_token:codec_start_token+codec_span_tokens]).astype(np.int64))
                prompt_slice = torch.from_numpy((codec[prompt_start_token:prompt_start_token+prompt_span_tokens]).astype(np.int64))
                frame_codes = flat_tokens_to_frame_codes(
                    codec_slice.view(frame_window, num_codebooks),
                    num_codebooks=num_codebooks,
                    codebook_size=int(meta['codec_vocab_size']),
                )
                coarse_rows.append(frame_codes[:, 0])
                residual_rows.append(frame_codes[:, 1:])
                prompt_codes = flat_tokens_to_frame_codes(
                    prompt_slice.view(prompt_frame_count, num_codebooks),
                    num_codebooks=num_codebooks,
                    codebook_size=int(meta['codec_vocab_size']),
                )
                keep_prompt_frames = prompt_frame_count
                if split == 'train' and style_prompt_dropout_prob > 0:
                    if torch.rand(1).item() < style_prompt_dropout_prob:
                        if torch.rand(1).item() < style_prompt_zero_prob:
                            keep_prompt_frames = 0
                        else:
                            min_keep = min(prompt_frame_count, max(1, int(style_prompt_min_frames)))
                            if min_keep < prompt_frame_count:
                                keep_prompt_frames = int(torch.randint(min_keep, prompt_frame_count + 1, (1,)).item())
                if keep_prompt_frames < prompt_frame_count:
                    prompt_codes = prompt_codes.clone()
                    prompt_prosody_slice = prompt_prosody_slice.clone()
                    prompt_codes[keep_prompt_frames:, :] = 0
                    prompt_prosody_slice[keep_prompt_frames:, :] = 0
                prompt_style_rows.append(prompt_codes)
                prompt_prosody_rows.append(prompt_prosody_slice)
                prompt_length_rows.append(torch.tensor(keep_prompt_frames, dtype=torch.long))
            x = (
                torch.stack(semantic_rows),
                torch.stack(coarse_rows),
                torch.stack(prompt_style_rows),
                torch.stack(prompt_prosody_rows),
                torch.stack(prompt_length_rows),
            )
            y = torch.stack(residual_rows)
        else:
            raise ValueError(f"Unsupported model_type={model_type!r} for dataset_type={dataset_type!r}")
    if dataset_type == 'audio_codec' and model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
        data = load_audio_frame_data(split)
        num_codebooks = int(meta['num_codebooks'])
        codebook_size = int(meta['codebook_size'])
        frame_count = len(data) // num_codebooks
        if frame_count <= block_size:
            raise ValueError(
                f"Dataset split {split} only has {frame_count} audio frames, smaller than block_size={block_size}"
            )
        frame_ix = torch.randint(frame_count - block_size, (batch_size,))
        token_ix = frame_ix * num_codebooks
        frame_span = (block_size + 1) * num_codebooks
        sequences = torch.stack(
            [torch.from_numpy((data[i:i+frame_span]).astype(np.int64)) for i in token_ix.tolist()]
        )
        frame_codes = flat_tokens_to_frame_codes(
            sequences.view(batch_size, block_size + 1, num_codebooks),
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )
        x = frame_codes[:, :block_size, :]
        y = frame_codes[:, 1:, :]
    elif dataset_type != 'audio_semantic_codec_v5':
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=data_bin_dtype, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=data_bin_dtype, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        if isinstance(x, tuple):
            x = tuple(part.pin_memory().to(device, non_blocking=True) for part in x)
        else:
            x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        if isinstance(x, tuple):
            x = tuple(part.to(device) for part in x)
        else:
            x = x.to(device)
        y = y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
last_eval_train_loss = None
last_eval_val_loss = None

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
meta = {}
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size')
    if 'data_dtype' in meta:
        data_bin_dtype = np.dtype(meta['data_dtype'])
    semantic_bin_dtype = np.dtype(meta.get('semantic_dtype', data_bin_dtype))
    codec_bin_dtype = np.dtype(meta.get('codec_dtype', data_bin_dtype))
    if 'stoi' in meta and 'itos' in meta:
        stoi, itos = meta['stoi'], meta['itos']
        tokenizer = Tokenizer(stoi=stoi, itos=itos)
        encode = tokenizer.encode
        decode = tokenizer.decode
    if meta_vocab_size is not None:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

dataset_type = meta.get('dataset_type', 'text')
if dataset_type == 'audio_codec' and model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size * int(meta['num_codebooks'])
    )
    print(f"audio-frame tokens per iteration will be: {tokens_per_iter:,}")
if dataset_type == 'audio_semantic_codec_v5':
    if model_type == 'audio_semantic_gpt':
        tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
        print(f"semantic tokens per iteration will be: {tokens_per_iter:,}")
    elif model_type == 'audio_coarse_decoder':
        tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
        print(f"coarse decoder tokens per iteration will be: {tokens_per_iter:,}")
    elif model_type == 'audio_fine_decoder':
        tokens_per_iter = (
            gradient_accumulation_steps
            * ddp_world_size
            * batch_size
            * block_size
            * max(1, int(meta['num_codebooks']) - 1)
        )
        print(f"fine decoder tokens per iteration will be: {tokens_per_iter:,}")
    elif model_type == 'audio_acoustic_decoder':
        tokens_per_iter = (
            gradient_accumulation_steps
            * ddp_world_size
            * batch_size
            * block_size
            * int(meta['num_codebooks'])
        )
        print(f"acoustic decoder tokens per iteration will be: {tokens_per_iter:,}")
audio_sampling_enabled = dataset_type == 'audio_codec' and bool(sample_prompt_audio)
audio_sample_dir = os.path.join(out_dir, sample_output_subdir)
audio_codec_model = None
audio_prompt_tokens = None
audio_prompt_codes = None
audio_prompt_frames = None
audio_prompt_wav = None


def maybe_trim_waveform(wav, sample_rate, max_seconds):
    if max_seconds <= 0:
        return wav
    max_samples = int(sample_rate * max_seconds)
    if wav.size(-1) <= max_samples:
        return wav
    return wav[..., :max_samples]


def sanitize_audio_tokens(tokens, separator_token_id):
    if separator_token_id is None:
        return tokens
    for idx, token in enumerate(tokens):
        if token == separator_token_id:
            return tokens[:idx]
    return tokens


def create_model(model_kind, args):
    args = dict(args)
    if model_kind == 'audio_frame_gpt':
        config = AudioFrameGPTConfig(**args)
        return AudioFrameGPT(config)
    if model_kind == 'audio_coarse_fine_gpt':
        config = AudioCoarseFineGPTConfig(**args)
        return AudioCoarseFineGPT(config)
    if model_kind == 'audio_semantic_gpt':
        config = AudioSemanticGPTConfig(**args)
        return AudioSemanticGPT(config)
    if model_kind == 'audio_coarse_decoder':
        config = AudioCoarseDecoderConfig(**args)
        return AudioCoarseDecoder(config)
    if model_kind == 'audio_fine_decoder':
        config = AudioFineDecoderConfig(**args)
        return AudioFineDecoder(config)
    if model_kind == 'audio_acoustic_decoder':
        config = AudioAcousticDecoderConfig(**args)
        return AudioAcousticDecoder(config)
    if model_kind == 'gpt':
        config = GPTConfig(**args)
        return GPT(config)
    raise ValueError(f"Unsupported model_type={model_kind!r}")


def model_forward(model_obj, X, Y):
    if model_type == 'audio_coarse_decoder':
        semantic_tokens, prompt_codec_frames, prompt_prosody_features, prompt_valid_lengths = X
        return model_obj(
            semantic_tokens,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            coarse_targets=Y,
        )
    if model_type == 'audio_fine_decoder':
        semantic_tokens, coarse_tokens, prompt_codec_frames, prompt_prosody_features, prompt_valid_lengths = X
        return model_obj(
            semantic_tokens,
            coarse_tokens,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            residual_targets=Y,
        )
    if model_type == 'audio_acoustic_decoder':
        semantic_tokens, prompt_codec_frames, prompt_prosody_features, prompt_valid_lengths = X
        return model_obj(
            semantic_tokens,
            prompt_codec_frames=prompt_codec_frames,
            prompt_prosody_features=prompt_prosody_features,
            prompt_valid_lengths=prompt_valid_lengths,
            codec_targets=Y,
        )
    return model_obj(X, Y)


@torch.no_grad()
def save_audio_progress_sample(step, model_for_sampling):
    global audio_codec_model, audio_prompt_tokens, audio_prompt_codes, audio_prompt_frames, audio_prompt_wav

    if not audio_sampling_enabled:
        return None

    if audio_codec_model is None:
        audio_codec_model = load_encodec_model(
            model_name=meta['model_name'],
            bandwidth=meta['bandwidth'],
            device='cpu',
        )

    if audio_prompt_tokens is None and audio_prompt_frames is None:
        prompt_wav, prompt_sample_rate = torchaudio.load(sample_prompt_audio)
        prompt_wav = maybe_trim_waveform(prompt_wav, prompt_sample_rate, sample_prompt_seconds)
        prompt_batch = encode_waveform(
            audio_codec_model,
            prompt_wav,
            prompt_sample_rate,
            device='cpu',
            codebook_size=meta['codebook_size'],
        )
        audio_prompt_codes = prompt_batch.codes
        audio_prompt_wav = (
            prompt_batch.normalized_wav
            if prompt_batch.normalized_wav is not None
            else decode_codes(audio_codec_model, audio_prompt_codes, device='cpu')
        )
        if model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
            audio_prompt_frames = prompt_batch.codes.transpose(0, 1).contiguous()
            prompt_frames = audio_prompt_frames.size(0)
            frames_per_second = meta['tokens_per_second'] / meta['num_codebooks']
            if prompt_frames > raw_model.config.block_size:
                prompt_seconds = prompt_frames / frames_per_second
                visible_seconds = raw_model.config.block_size / frames_per_second
                print(
                    f"warning: audio sample prompt is {prompt_seconds:.2f}s but frame block_size only "
                    f"covers {visible_seconds:.2f}s; generation is conditioned on the tail only"
                )
        else:
            audio_prompt_tokens = prompt_batch.flattened_tokens.tolist()
            if len(audio_prompt_tokens) > raw_model.config.block_size:
                prompt_seconds = len(audio_prompt_tokens) / meta['tokens_per_second']
                visible_seconds = raw_model.config.block_size / meta['tokens_per_second']
                print(
                    f"warning: audio sample prompt is {prompt_seconds:.2f}s but block_size only "
                    f"covers {visible_seconds:.2f}s; generation is conditioned on the tail only"
                )

    num_codebooks = meta['num_codebooks']
    codebook_size = meta['codebook_size']

    if model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
        frames_per_second = meta['tokens_per_second'] / num_codebooks
        max_new_frames = max(1, int(sample_max_new_seconds * frames_per_second))
        x = torch.tensor(audio_prompt_frames.tolist(), dtype=torch.long, device=device)[None, ...]
        if model_type == 'audio_coarse_fine_gpt':
            generated = generate_audio_coarse_fine_frames(
                model_for_sampling,
                x,
                max_new_frames=max_new_frames,
                temperature=sample_temperature,
                top_k=sample_top_k,
            )
        else:
            generated = generate_audio_frames(
                model_for_sampling,
                x,
                max_new_frames=max_new_frames,
                temperature=sample_temperature,
                top_k=sample_top_k,
            )
        generated_frames = generated[0]
        if generated_frames.numel() == 0:
            print(f"audio sample at step {step} produced no decodable frames")
            return None
        codes = generated_frames.transpose(0, 1).contiguous().cpu()
    else:
        tokens_per_second = meta['tokens_per_second']
        max_new_tokens = int(sample_max_new_seconds * tokens_per_second)
        separator_token_id = meta.get('separator_token_id')
        x = torch.tensor(audio_prompt_tokens, dtype=torch.long, device=device)[None, ...]
        generated = generate_audio_tokens(
            model_for_sampling,
            x,
            max_new_tokens=max_new_tokens,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            temperature=sample_temperature,
            top_k=sample_top_k,
        )

        all_tokens = sanitize_audio_tokens(generated[0].tolist(), separator_token_id)
        valid_token_count = len(all_tokens) - (len(all_tokens) % num_codebooks)
        all_tokens = all_tokens[:valid_token_count]
        if not all_tokens:
            print(f"audio sample at step {step} produced no decodable tokens")
            return None
        codes = unflatten_codes(
            torch.tensor(all_tokens, dtype=torch.long),
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
        )
    wav = decode_codes(audio_codec_model, codes, device='cpu')
    os.makedirs(audio_sample_dir, exist_ok=True)
    prompt_wav = audio_prompt_wav
    prompt_num_samples = prompt_wav.size(-1)
    continuation_wav = wav[:, prompt_num_samples:]

    sample_path = os.path.join(audio_sample_dir, f"step_{step:06d}_full.wav")
    continuation_path = os.path.join(audio_sample_dir, f"step_{step:06d}_continuation.wav")
    prompt_path = os.path.join(audio_sample_dir, f"step_{step:06d}_prompt.wav")
    save_waveform(prompt_path, prompt_wav, meta['sample_rate'])
    save_waveform(sample_path, wav, meta['sample_rate'])
    save_waveform(continuation_path, continuation_wav, meta['sample_rate'])
    print(f"saved audio sample to {sample_path}")
    print(f"saved continuation sample to {continuation_path}")
    return {
        'prompt_wav': prompt_wav,
        'full_wav': wav,
        'continuation_wav': continuation_wav,
        'sample_rate': meta['sample_rate'],
        'prompt_path': prompt_path,
        'full_path': sample_path,
        'continuation_path': continuation_path,
    }

# model init
if model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
    if dataset_type != 'audio_codec':
        raise ValueError(f"{model_type} requires an audio_codec dataset")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
        num_codebooks=int(meta['num_codebooks']),
        codebook_size=int(meta['codebook_size']),
    )
elif model_type == 'audio_semantic_gpt':
    if dataset_type != 'audio_semantic_codec_v5':
        raise ValueError("audio_semantic_gpt requires an audio_semantic_codec_v5 dataset")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
        vocab_size=int(meta['semantic_vocab_size']),
    )
elif model_type == 'audio_coarse_decoder':
    if dataset_type != 'audio_semantic_codec_v5':
        raise ValueError("audio_coarse_decoder requires an audio_semantic_codec_v5 dataset")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
        semantic_vocab_size=int(meta['semantic_vocab_size']),
        codec_vocab_size=int(meta['codec_vocab_size']),
        semantic_to_codec_ratio=int(meta['semantic_to_codec_ratio']),
        style_prompt_frames=style_prompt_frames,
    )
elif model_type == 'audio_fine_decoder':
    if dataset_type != 'audio_semantic_codec_v5':
        raise ValueError("audio_fine_decoder requires an audio_semantic_codec_v5 dataset")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
        semantic_vocab_size=int(meta['semantic_vocab_size']),
        codec_vocab_size=int(meta['codec_vocab_size']),
        num_codebooks=int(meta['num_codebooks']),
        semantic_to_codec_ratio=int(meta['semantic_to_codec_ratio']),
        style_prompt_frames=style_prompt_frames,
    )
elif model_type == 'audio_acoustic_decoder':
    if dataset_type != 'audio_semantic_codec_v5':
        raise ValueError("audio_acoustic_decoder requires an audio_semantic_codec_v5 dataset")
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
        semantic_vocab_size=int(meta['semantic_vocab_size']),
        codec_vocab_size=int(meta['codec_vocab_size']),
        num_codebooks=int(meta['num_codebooks']),
        semantic_to_codec_ratio=int(meta['semantic_to_codec_ratio']),
        style_prompt_frames=style_prompt_frames,
    )
else:
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if model_type == 'gpt' and meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    if model_type == 'gpt':
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model = create_model(model_type, model_args)
elif init_from == 'resume':
    ckpt_path = resume_ckpt_path or os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    # resume training from a checkpoint.
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_type = checkpoint.get('model_type', checkpoint.get('config', {}).get('model_type', model_type))
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    if model_type in {'audio_frame_gpt', 'audio_coarse_fine_gpt'}:
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout', 'num_codebooks', 'codebook_size']:
            model_args[k] = checkpoint_model_args[k]
    elif model_type == 'audio_semantic_gpt':
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
    elif model_type == 'audio_coarse_decoder':
        for k in [
            'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout',
            'semantic_vocab_size', 'codec_vocab_size', 'semantic_to_codec_ratio',
            'style_prompt_frames'
        ]:
            model_args[k] = checkpoint_model_args[k]
    elif model_type == 'audio_fine_decoder':
        for k in [
            'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout',
            'semantic_vocab_size', 'codec_vocab_size', 'num_codebooks', 'semantic_to_codec_ratio',
            'style_prompt_frames'
        ]:
            model_args[k] = checkpoint_model_args[k]
    elif model_type == 'audio_acoustic_decoder':
        for k in [
            'n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout',
            'semantic_vocab_size', 'codec_vocab_size', 'num_codebooks', 'semantic_to_codec_ratio',
            'style_prompt_frames'
        ]:
            model_args[k] = checkpoint_model_args[k]
    else:
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    model = create_model(model_type, model_args)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = 0 if reset_iteration_on_resume else checkpoint['iter_num']
    best_val_loss = 1e9 if reset_best_val_loss_on_resume else checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if model_type != 'gpt':
        raise ValueError("OpenAI GPT-2 init is only supported for model_type='gpt'")
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    if not reset_optimizer_on_resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model_forward(model, X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Guard short debug runs where decay window is collapsed or invalid.
    if lr_decay_iters <= warmup_iters:
        return min_lr
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

tb_writer = None
if tensorboard_log and master_process:
    from torch.utils.tensorboard import SummaryWriter

    tb_name = tensorboard_run_name or Path(out_dir).name
    tb_dir = os.path.join(out_dir, 'tensorboard', tb_name)
    tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=tensorboard_flush_secs)
    tb_writer.add_text('config/json', json.dumps(config, indent=2, sort_keys=True), 0)
    print(f"tensorboard logging to {tb_dir}")

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

stop_flag = False
save_flag = False
save_asap_flag = False
save_checkpoint = False

def key_listener():
    global stop_flag, save_flag, save_asap_flag, save_checkpoint
    while True:
        key = sys.stdin.read(1).lower()  # blocking call, waits for 1 char
        if key == 's':
            stop_flag = True
            break
        if key == 'd':
            save_flag = True
        if key == 'a':
            save_asap_flag = True
        if key == 'c':
            save_checkpoint = True


def build_checkpoint_payload():
    return {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_type': model_type,
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'last_eval_train_loss': last_eval_train_loss,
        'last_eval_val_loss': last_eval_val_loss,
        'config': config,
    }


def step_checkpoint_dir():
    return Path(out_dir) / 'checkpoints'


def step_checkpoint_path(step: int) -> Path:
    return step_checkpoint_dir() / f'ckpt_step_{step:06d}.pt'


def prune_step_checkpoints():
    if checkpoint_keep_last <= 0:
        return
    checkpoint_dir = step_checkpoint_dir()
    if not checkpoint_dir.is_dir():
        return
    paths = sorted(checkpoint_dir.glob('ckpt_step_*.pt'))
    stale = paths[:-checkpoint_keep_last]
    for path in stale:
        try:
            path.unlink()
        except OSError:
            pass


def persist_checkpoint(payload: dict, save_snapshot: bool) -> None:
    main_path = Path(out_dir) / 'ckpt.pt'
    print(f"saving checkpoint to {main_path}")
    torch.save(payload, main_path)
    if save_snapshot:
        snapshot_path = step_checkpoint_path(iter_num)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"saving step checkpoint to {snapshot_path}")
        torch.save(payload, snapshot_path)
        prune_step_checkpoints()

# start listener in background
threading.Thread(target=key_listener, daemon=True).start()

print("Starting training loop, press 's' to stop, 'd' to save checkpoint, 'a' to save on next iteration")
val_data = {}
prev_val_data = {}  # Initialize dictionary to store previous validation results

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        last_eval_train_loss = float(losses['train'])
        last_eval_val_loss = float(losses['val'])
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if tb_writer is not None:
            tb_writer.add_scalar('eval/train_loss', losses['train'], iter_num)
            tb_writer.add_scalar('eval/val_loss', losses['val'], iter_num)
            tb_writer.add_scalar('train/lr', lr, iter_num)
            if running_mfu >= 0:
                tb_writer.add_scalar('train/mfu_percent', running_mfu * 100, iter_num)
        audio_sample = save_audio_progress_sample(iter_num, raw_model)
        if tb_writer is not None and audio_sample is not None:
            tb_writer.add_audio(
                'audio/prompt',
                audio_sample['prompt_wav'],
                iter_num,
                sample_rate=audio_sample['sample_rate'],
            )
            tb_writer.add_audio(
                'audio/full',
                audio_sample['full_wav'],
                iter_num,
                sample_rate=audio_sample['sample_rate'],
            )
            tb_writer.add_audio(
                'audio/continuation',
                audio_sample['continuation_wav'],
                iter_num,
                sample_rate=audio_sample['sample_rate'],
            )
        # Optional text-only sampling hook for arithmetic-style datasets.
        response = ""
        if 'encode' in globals() and 'decode' in globals():
            try:
                with torch.no_grad():
                    start_tokens = encode('<')
                    end_tokens = encode('>')
                    if isinstance(start_tokens, int):
                        start_tokens = [start_tokens]
                    if isinstance(end_tokens, int):
                        end_tokens = [end_tokens]
                    start_token = torch.tensor(start_tokens, device=device, dtype=torch.long)[None, :]
                    logits = model.generate(start_token, max_new_tokens=block_size, end_tokens=end_tokens)
                    response = decode(logits[0].tolist())
                    print(response)

                    with open('evolution.txt', 'a') as f:
                        f.write(response + '\n')
            except Exception:
                # Standard text datasets don't always define these markers; skip quietly.
                response = ""
        if iter_num % eval_interval_extra == 0 and master_process and False:
            pass
        save_snapshot_now = checkpoint_interval > 0 and iter_num > 0 and iter_num % checkpoint_interval == 0
        if losses['val'] < best_val_loss or always_save_checkpoint or save_flag or save_snapshot_now:
            save_flag = False
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = build_checkpoint_payload()
                persist_checkpoint(checkpoint, save_snapshot=save_snapshot_now)
    if save_asap_flag and master_process:
        save_asap_flag = False
        checkpoint = build_checkpoint_payload()
        persist_checkpoint(checkpoint, save_snapshot=False)
    if stop_flag:
        print("Stopping training...")
        break
    if save_checkpoint:
        save_checkpoint = False
        checkpoint = build_checkpoint_payload()
        persist_checkpoint(checkpoint, save_snapshot=False)


    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model_forward(model, X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if tb_writer is not None:
            tb_writer.add_scalar('train/iter_loss', lossf, iter_num)
            tb_writer.add_scalar('train/iter_time_ms', dt * 1000, iter_num)
            if running_mfu >= 0:
                tb_writer.add_scalar('train/mfu_percent', running_mfu * 100, iter_num)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()


if wandb_log and master_process and wandb_model_upload:
    # Load the best checkpoint
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        print(f"Uploading best checkpoint from {checkpoint_path} and meta to wandb")
        artifact = wandb.Artifact(f'model-{wandb.run.id}', type='model')
        artifact.add_file(checkpoint_path)
        artifact.add_file(meta_path)
        wandb.log_artifact(artifact)
        print("Checkpoint and meta uploaded to wandb successfully")
    wandb.finish()

if tb_writer is not None:
    tb_writer.flush()
    tb_writer.close()

print("bye")
