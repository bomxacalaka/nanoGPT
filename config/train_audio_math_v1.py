# fine-tune the current speech model on a tiny synthetic spoken-math dataset

out_dir = "out-audio-math-v1"
eval_interval = 5000
eval_interval_extra = 50000
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
checkpoint_interval = 5000
checkpoint_keep_last = 24
compile = True

wandb_log = False
wandb_project = "audio-math"
wandb_run_name = "audio_math_v1"
tensorboard_log = True
tensorboard_run_name = "audio_math_v1"
console_log_file = "train.log"

dataset = "audio_math_codec"
gradient_accumulation_steps = 4
batch_size = 8
block_size = 2048

n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

init_from = "resume"
resume_ckpt_path = "out-audio-librispeech-v2/ckpt.pt"
reset_optimizer_on_resume = True
reset_iteration_on_resume = True
reset_best_val_loss_on_resume = True

learning_rate = 1e-4
max_iters = 200000
lr_decay_iters = 200000
min_lr = 1e-5
beta2 = 0.95
warmup_iters = 500

sample_prompt_audio = "data/audio_math_codec/raw/val/000_00_00_00.wav"
sample_prompt_seconds = 0.9
sample_max_new_seconds = 2.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
