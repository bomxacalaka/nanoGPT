import pickle
import json

# Load the pickle file
with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)

# Create tokenizer configuration
tokenizer = {
    'vocab_size': meta['vocab_size'],
    'vocab': {
        'stoi': meta['stoi'],  # string to integer mapping
        'itos': meta['itos'],  # integer to string mapping
    }
}

# Save as JSON
with open('out-shakespeare-char/tokenizer.json', 'w') as f:
    json.dump(tokenizer, f, indent=2)

print("Tokenizer configuration saved to tokenizer.json")
