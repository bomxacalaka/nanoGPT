import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from tokenizer import Tokenizer
from colour_print import cprint

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 5 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
# Filter out Jupyter notebook arguments before executing configurator
import sys
original_argv = sys.argv.copy()
sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]
exec(open('configurator.py').read()) # overrides from command line or config file
sys.argv = original_argv  # restore original arguments
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    tokenizer = Tokenizer(stoi=stoi, itos=itos)
    encode = tokenizer.encode
    decode = tokenizer.decode
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f"Val loss: {checkpoint['best_val_loss'].item()}")


import torch
from itertools import product

# === prompt and model inference ===
def prompt_it(prompt):
    x = encode(prompt)
    x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]
    y = model.generate(x, 100, temperature=0.1, top_k=top_k, end_tokens=encode('>'))
    response = decode(y[0].tolist())
    return response

# === number to words (0–999) ===
def number_to_words(n):
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    if n < 20:
        return ones[n]
    elif n < 100:
        t = n // 10
        o = n % 10
        return tens[t] if o == 0 else f"{tens[t]}-{ones[o]}"
    else:
        h = n // 100
        r = n % 100
        if r == 0:
            return f"{ones[h]} hundred"
        else:
            return f"{ones[h]} hundred {number_to_words(r)}"

# === words to number (0–999) ===
def words_to_number(words):
    words = words.replace("-", " ")
    tokens = words.split()
    num = 0
    temp = 0
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    for w in tokens:
        if w == "hundred":
            temp *= 100
        elif w in word_to_num:
            temp += word_to_num[w]
    num += temp
    return num

# === expression verification ===
def verify_arithmetic_expression(expression):
    expr = expression.strip().replace(">", "").replace("<", "")
    parts = expr.split()

    try:
        if "plus" in parts:
            op = "plus"
        elif "minus" in parts:
            op = "minus"
        elif "times" in parts:
            op = "times"
        elif "divided" in parts:
            op = "divided"
        else:
            return False, "No recognized operator"

        eq_index = parts.index("equals")
        op_index = parts.index(op)

        num1 = words_to_number(" ".join(parts[:op_index]))
        num2 = words_to_number(" ".join(parts[op_index + 1:eq_index]))
        result = words_to_number(" ".join(parts[eq_index + 1:]))

        if op == "plus":
            actual = num1 + num2
        elif op == "minus":
            actual = num1 - num2
        elif op == "times":
            actual = num1 * num2
        elif op == "divided":
            if num2 == 0:
                return False, "Division by zero"
            actual = num1 // num2 if num1 % num2 == 0 else None

        if actual is None:
            return False, "Non-integer division"

        return (actual == result), f"{num1} {op} {num2} = {actual}, got {result}"
    except Exception as e:
        return False, f"Error parsing: {e}"

from itertools import product

val = []
operations = ["plus", "minus", "times", "divided by"]

for i, j in product(range(99), range(99)):
    for op in operations:
        # apply same validity rules as your test loop
        if op == "plus":
            valid = (i + j <= 999)
            if not valid: 
                continue
            result = i + j
        elif op == "minus":
            valid = (i >= j)
            if not valid:
                continue
            result = i - j
        elif op == "times":
            valid = (i * j <= 999)
            if not valid:
                continue
            result = i * j
        elif op == "divided by":
            valid = (j != 0 and i % j == 0 and (i // j) <= 999)
            if not valid:
                continue
            result = i // j
        # build the verbal test string
        val.append(f"<{number_to_words(i)} {op} {number_to_words(j)} equals")

from collections import defaultdict

grouped = defaultdict(list)

for expr in val:
    length = len(expr.split())
    grouped[length].append(expr)

# turn it into a list of groups sorted by length
grouped_val = [grouped[k] for k in sorted(grouped.keys())]

def prompt_it(prompts):
   # Handle both single prompt (string) and multiple prompts (list)
   if isinstance(prompts, str):
       prompts = [prompts]
   
   # Encode all prompts
   encoded = [torch.tensor(encode(prompt), dtype=torch.long) for prompt in prompts]
   
   # Pad sequences to same length
   min_len = min(len(t) for t in encoded)
   encoded = [t[:min_len] for t in encoded]
   
   # Use detach().clone() to avoid the warning
   x = torch.stack([t.detach().clone() for t in encoded]).to(device)
   
   # Generate for all prompts in batch
   with torch.no_grad():
       with ctx:
           special_end_token = encode('>')
           y = model.generate(x, 60, temperature=0.8, top_k=top_k, end_tokens=special_end_token)
           
   # Decode all outputs
   responses = []
   for sample in y:
       responses.append(decode(sample.tolist()).split('>')[0].replace('<', ''))
   
   # Return single response if input was single prompt, otherwise return list
   return responses[0] if len(responses) == 1 else responses

def process_lists(input_list):
    result = []
    for item in input_list:
        # Split each list into chunks of 1000 items
        for i in range(0, len(item), 4000):
            result.append(item[i:i+4000])
    return result

correct_count = 0
total_count = 0
log_every = 5000

grouped_val = process_lists(grouped_val)

from tqdm import tqdm

group_pbar = tqdm(grouped_val, desc="Evaluating groups", unit="group")

for group in group_pbar:
    result = prompt_it(group)
    expr_pbar = tqdm(result, desc="Verifying", leave=False, unit="expr")
    
    for expression in expr_pbar:
        is_correct, msg = verify_arithmetic_expression(expression)
        total_count += 1
        if is_correct:
            correct_count += 1
        acc = correct_count / total_count * 100 if total_count > 0 else 0
        # Update postfix with accuracy on expression progress bar
        expr_pbar.set_postfix({"Accuracy": f"{acc:.2f}%"})
    
    # Update group progress bar with final accuracy for that group
    group_pbar.set_postfix({"Accuracy": f"{acc:.2f}%"})

