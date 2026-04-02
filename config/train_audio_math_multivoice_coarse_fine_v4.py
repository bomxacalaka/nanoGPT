# v4 math fine-tune: coarse-stream AR plus parallel residual-codebook prediction

out_dir = "out-audio-math-multivoice-coarse-fine-v4"
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
wandb_run_name = "audio_math_multivoice_coarse_fine_v4"
tensorboard_log = True
tensorboard_run_name = "audio_math_multivoice_coarse_fine_v4"
console_log_file = "train.log"

dataset = "audio_math_multivoice_large_codec"
model_type = "audio_coarse_fine_gpt"
gradient_accumulation_steps = 4
batch_size = 12
block_size = 256

n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.1

init_from = "resume"
resume_ckpt_path = "out-audio-librispeech-coarse-fine-v4/ckpt.pt"
reset_optimizer_on_resume = True
reset_iteration_on_resume = True
reset_best_val_loss_on_resume = True

learning_rate = 8e-5
max_iters = 200000
lr_decay_iters = 200000
min_lr = 8e-6
beta2 = 0.95
warmup_iters = 500

sample_prompt_audio = "data/audio_math_multivoice_large_codec/raw/val/000000_t00_EN-AU_00_00_00.wav"
sample_prompt_seconds = 0.9
sample_max_new_seconds = 2.0
sample_temperature = 0.8
sample_top_k = 50
sample_output_subdir = "samples"
