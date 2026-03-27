# stronger LibriSpeech continuation baseline with more usable context

out_dir = "out-audio-librispeech-v2"
eval_interval = 200
eval_interval_extra = 2000
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
compile = True

wandb_log = False
wandb_project = "audio-librispeech"
wandb_run_name = "audio_librispeech_v2"

dataset = "audio_librispeech_codec"
gradient_accumulation_steps = 4
batch_size = 8
block_size = 2048

n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 2e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 2e-5
beta2 = 0.95
warmup_iters = 200

sample_prompt_audio = "data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2/174/168635/174-168635-0002.flac"
sample_prompt_seconds = 2.0
sample_max_new_seconds = 1.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
