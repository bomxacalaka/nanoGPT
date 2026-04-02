# V5 semantic autoregressive speech model

out_dir = "out-audio-semantic-v5"
eval_interval = 500
eval_iters = 20
log_interval = 10
init_from = "resume"

always_save_checkpoint = False
checkpoint_interval = 2000
checkpoint_keep_last = 1
compile = True

wandb_log = False
tensorboard_log = False
console_log_file = "train.log"

dataset = "audio_semantic_codec"
model_type = "audio_semantic_gpt"

gradient_accumulation_steps = 4
batch_size = 16
block_size = 512

n_layer = 16
n_head = 16
n_embd = 256*2
dropout = 0.3

learning_rate = 2e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 2e-5
beta2 = 0.95
warmup_iters = 200
