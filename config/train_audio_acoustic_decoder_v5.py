# V5 acoustic decoder conditioned on semantic tokens

out_dir = "out-audio-acoustic-decoder-v5"
eval_interval = 500
eval_iters = 20
log_interval = 10
init_from = "scratch"

always_save_checkpoint = False
checkpoint_interval = 2000
checkpoint_keep_last = 1
compile = True

wandb_log = False
tensorboard_log = False
console_log_file = "train.log"

dataset = "audio_semantic_codec"
model_type = "audio_acoustic_decoder"
style_prompt_frames = 96
style_prompt_dropout_prob = 0.5
style_prompt_zero_prob = 0.1
style_prompt_min_frames = 8

gradient_accumulation_steps = 4
batch_size = 8
block_size = 384

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.1

learning_rate = 2e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 2e-5
beta2 = 0.95
warmup_iters = 200
