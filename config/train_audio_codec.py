# train a small GPT on flattened EnCodec token streams

out_dir = "out-audio-codec"
eval_interval = 500
eval_interval_extra = 2000
eval_iters = 100
log_interval = 10

always_save_checkpoint = False
compile = True

wandb_log = False
wandb_project = "audio-codec"
wandb_run_name = "audio_codec_tiny"

dataset = "audio_codec"
gradient_accumulation_steps = 4
batch_size = 8
block_size = 1024

n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 200

sample_prompt_audio = "data/audio_codec/raw/free-spoken-digit-dataset/recordings/3_nicolas_7.wav"
sample_max_new_seconds = 0.5
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
