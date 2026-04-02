# frame-level EnCodec continuation model: one transformer step per audio frame,
# predicting all codebooks in parallel instead of flattening them into one long stream

out_dir = "out-audio-librispeech-frame-v3"
eval_interval = 500
eval_interval_extra = 5000
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
checkpoint_interval = 5000
checkpoint_keep_last = 24
compile = True

wandb_log = False
wandb_project = "audio-librispeech"
wandb_run_name = "audio_librispeech_frame_v3"
tensorboard_log = True
tensorboard_run_name = "audio_librispeech_frame_v3"
console_log_file = "train.log"

dataset = "audio_librispeech_codec"
model_type = "audio_frame_gpt"
gradient_accumulation_steps = 4
batch_size = 12
block_size = 256

n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 2e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 2e-5
beta2 = 0.95
warmup_iters = 500

sample_prompt_audio = "data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2/174/168635/174-168635-0002.flac"
sample_prompt_seconds = 2.0
sample_max_new_seconds = 3.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
