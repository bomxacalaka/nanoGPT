# v4: autoregressive only on the coarse EnCodec stream, predict residual codebooks in parallel

out_dir = "out-audio-librispeech-coarse-fine-v4"
eval_interval = 2000
eval_interval_extra = 5000
eval_iters = 20
log_interval = 10
init_from = "scratch"

always_save_checkpoint = False
checkpoint_interval = 5000
checkpoint_keep_last = 24
compile = True

wandb_log = False
wandb_project = "audio-librispeech"
wandb_run_name = "audio_librispeech_coarse_fine_v4"
tensorboard_log = True
tensorboard_run_name = "audio_librispeech_coarse_fine_v4"
console_log_file = "train.log"

dataset = "audio_librispeech_codec"
model_type = "audio_coarse_fine_gpt"
gradient_accumulation_steps = 8
batch_size = 16
block_size = 256*4

n_layer = 8*2
n_head = 8*2
n_embd = 256*2
dropout = 0.3

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
