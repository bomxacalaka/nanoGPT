# stronger LibriSpeech continuation baseline with more usable context

out_dir = "out-audio-librispeech-v2"
eval_interval = 200
eval_interval_extra = 2000
eval_iters = 20
log_interval = 10
init_from = "scratch"

always_save_checkpoint = False
checkpoint_interval = 2000
checkpoint_keep_last = 10
compile = True

wandb_log = False
wandb_project = "audio-librispeech"
wandb_run_name = "audio_librispeech_v2"
tensorboard_log = True
tensorboard_run_name = "audio_librispeech_v2"
console_log_file = "train.log"

dataset = "audio_librispeech_codec"
gradient_accumulation_steps = 4
batch_size = 8
block_size = 2048*2

n_layer = 16
n_head = 12
n_embd = 256*2
dropout = 0.3

learning_rate = 2e-4
max_iters = 20000
lr_decay_iters = 5000
min_lr = 2e-5
beta2 = 0.95
warmup_iters = 200

sample_prompt_audio = "data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2/174/168635/174-168635-0002.flac"
sample_prompt_seconds = 1.0
sample_max_new_seconds = 3.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
