import wandb
from simplevibe import llama3
import os

# input_file_path = os.path.join(os.path.dirname(__file__), 'data', 'shakespeare_char', '0_9.txt')
# with open(input_file_path, 'r') as f:
#     messages = f.read()
# try:
#     wandb_run_name = llama3(messages=[
#         {"role": "system", "content": "You are an expert in naming an AI model run based on what it is being trained on.\nYou will be given a sample of the data and from that you must generate a suitable name for the model. You MUST only return the name and nothing else.\nBe aware it's only a sample, so don't be too specific."},
#         {"role": "user", "content": f"Here is a sample of the data:\n{messages[:5000]}"}
#     ])['output'].strip('"')
# except Exception as e:
#     print(f"Llama3 Down: {e}")
# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 1000 # keep frequent because we'll overfit
eval_interval_extra = 5000
eval_iters = 500
log_interval = 100 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False
compile = True

wandb_log = True # override via command line if you like
wandb_project = 'mini-models'
wandb_model_upload = True
wandb_run_name = 'vibe_maths'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 8
batch_size = 64
block_size = 128 # context of up to 100 previous characters
# init_from = 'resume'

# baby GPT model :)
n_layer = 6   # number of transformer blocks (depth of the model) GPT-2 small used 12, GPT-3 went 96+.
n_head  = 6    # number of attention heads in multi-head attention Usually, n_embd is divisible by n_head. GPT-2 small had 12 heads, GPT-3 175B had 96 heads.
n_embd  = 120  # embedding size (hidden dimension of each token vector) GPT-2 small used 768, BERT-base used 768, GPT-3 used 12,288.
dropout = 0.3 # probability of dropping units (regularization) GPT-style large models often use 0.1.

learning_rate  = 1e-3 # with baby networks can afford to go a bit higher
max_iters      = 100000
lr_decay_iters = 100000 # make equal to max_iters usually
min_lr         = 1e-5 # learning_rate / 10 usually
beta2          = 0.99 # make a bit bigger because number of tokens per iter is small

decay_lr = True

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

n_layer = int(n_layer)
n_head = int(n_head)
n_embd = int(n_embd)

# Check if a run with this name already exists in the project
try:
    api = wandb.Api()
    runs = list(api.runs(wandb_project))
    names = [run.name for run in runs]
    if wandb_run_name in names:
        wandb_run_name = f"{wandb_run_name}-{len(names)}"
except:
    pass