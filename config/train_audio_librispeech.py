# train a small GPT on flattened EnCodec token streams from Mini LibriSpeech

out_dir = "out-audio-librispeech"
eval_interval = 100
eval_interval_extra = 1000
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
compile = True

wandb_log = False
wandb_project = "audio-librispeech"
wandb_run_name = "audio_librispeech_tiny"

dataset = "audio_librispeech_codec"
gradient_accumulation_steps = 2
batch_size = 12
block_size = 512

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 3e-4
max_iters = 2000
lr_decay_iters = 2000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 50

sample_prompt_audio = "data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2/174/168635/174-168635-0002.flac"
sample_max_new_seconds = 1.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
