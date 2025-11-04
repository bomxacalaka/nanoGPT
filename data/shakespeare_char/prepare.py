"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tokenizer import Tokenizer
from time import time

# download the tiny shakespeare dataset
special_tokens = ['<|start|>', '<|answer|>', '<|end|>']
include_special_tokens = False # whether to include the parts that make up special tokens in the character set disable if your dataset doesnt contain any e.g no < | s t a r t | >
if len(sys.argv) > 1:
    input_file_path = os.path.join(os.path.dirname(__file__), sys.argv[1])
else:
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

tokenizer = Tokenizer(data=data, include_special_tokens=include_special_tokens)
start_time = time()
tokenizer.fit()
end_time = time()
print(f"Tokenizer fitting took {end_time - start_time:.2f} seconds.")

# create the train and test splits
# Find the position of the last complete record in the first 90% of data
split_point = int(len(data) * 0.99)
# Look for the next <|end|> token after this point
end_token = ">\n"
next_end = data.find(end_token, split_point)
if next_end != -1:
    # Split after the end token (including the newline)
    split_point = next_end + len(end_token)
else:
    # Fallback if no end token is found
    print("Warning: Couldn't find clean split point. Using approximate split.")

train_data = data[:split_point]
val_data = data[split_point:]

# Verify the split is clean
print(f"Train data ends with: {train_data[-20:].replace('\n', '\\n')}")
print(f"Val data begins with: {val_data[:20].replace('\n', '\\n')}")

# encode both to integers
start_time = time()
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
end_time = time()
print(f"Encoding took {end_time - start_time:.2f} seconds.")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

meta = {
    'vocab_size': tokenizer.vocab_size,
    'itos': tokenizer.itos,
    'stoi': tokenizer.stoi,
    'dataset': input_file_path
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


# Add previous dataset but only their meta
# download the tiny shakespeare dataset
# input_file_path = os.path.join(os.path.dirname(__file__), 'base.txt')

# with open(input_file_path, 'r') as f:
#     data = f.read()
# print(f"length of dataset in characters: {len(data):,}")

# # get all the unique characters that occur in this text
# chars = sorted(list(set(data)))
# vocab_size_2 = len(chars)
# print("all the unique characters:", ''.join(chars))
# print(f"vocab size: {vocab_size:,}")

# # create a mapping from characters to integers
# stoi_2 = { ch:i for i,ch in enumerate(chars) }
# itos_2 = { i:ch for i,ch in enumerate(chars) }

# # save the meta information as well, to help us encode/decode later
# meta = {
#     'vocab_size': vocab_size + vocab_size_2,
#     'itos': {**itos, **itos_2},
#     'stoi': {**stoi, **stoi_2},
# }
# with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
#     pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
