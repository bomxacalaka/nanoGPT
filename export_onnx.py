import torch
import pickle
from model import GPTConfig, GPT

def export_to_onnx():
    # Load tokenizer from meta.pkl
    with open('data/shakespeare_char/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    itos = meta['itos']  # integer to string mapping
    stoi = meta['stoi']  # string to integer mapping
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    print("Vocabulary:", itos)
    
    # Load your trained model
    ckpt_path = 'out-shakespeare-char/ckpt.pt'
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Create model configuration
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Fix state dict keys
    state_dict = checkpoint['model']
    # Remove "_orig_mod." prefix from keys if present (from torch.compile)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
    
    # Load the state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input with start token
    start_token = stoi['<|start|>']
    dummy_input = torch.tensor([[start_token]], dtype=torch.long)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'out-shakespeare-char/ckpt.onnx',
        opset_version=14,  # Updated to support scaled_dot_product_attention
        export_params=True,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence', 2: 'vocab'}
        }
    )
    print("Model exported to ONNX format successfully!")

if __name__ == '__main__':
    export_to_onnx()
