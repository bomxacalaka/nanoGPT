import torch
import pickle
from model import GPTConfig, GPT
import json
import wandb
import os
import sys

def _get_runtime_from_artifact(api, artifact):
    try:
        producer_run = artifact.logged_by()  # docs: returns the run that originally logged the artifact
        if producer_run:
            # producer_run may be a wandb Run-like object with .summary
            # Check common places where runtime appears:
            summary = getattr(producer_run, "summary", None) or {}
            # common keys seen in the wild:
            if isinstance(summary, dict):
                # direct _runtime (works in many cases)
                if '_runtime' in summary:
                    return summary.get('_runtime')
                # nested _wandb.runtime (recommended for finished runs)
                wandb_block = summary.get('_wandb')
                if isinstance(wandb_block, dict) and 'runtime' in wandb_block:
                    return wandb_block.get('runtime')

            # If producer_run is a public API object, you might need to fetch summary via Api.run
            # Try resolving a full run path and re-query via API for a more complete Run object:
            try:
                # artifact.logged_by() might give a Run object or just an id-like string
                run_id = None
                if hasattr(producer_run, "id"):
                    run_id = producer_run.id
                elif isinstance(producer_run, str):
                    run_id = producer_run
                if run_id:
                    run_path = f"{artifact.entity}/{artifact.project}/{run_id}"
                    api_run = api.run(run_path)
                    api_summary = getattr(api_run, "summary", None) or {}
                    if '_runtime' in api_summary:
                        return api_summary.get('_runtime')
                    wandb_block = api_summary.get('_wandb')
                    if isinstance(wandb_block, dict) and 'runtime' in wandb_block:
                        return wandb_block.get('runtime')
            except Exception:
                # ignore and move on
                pass
    except Exception:
        pass

    # Not found
    return None


def export_to_onnx(artifact_dir, artifact, api):
    meta_path = os.path.join(artifact_dir, 'meta.pkl')
    ckpt_path = os.path.join(artifact_dir, 'ckpt.pt')

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    vocab_size = meta['vocab_size']
    itos = meta['itos']
    stoi = meta['stoi']
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_args = checkpoint['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    start_token = stoi.get('<', 0)
    dummy_input = torch.tensor([[start_token]], dtype=torch.long)

    os.makedirs('output', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        'output/model.onnx',
        opset_version=14,
        export_params=True,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'logits': {0: 'batch', 1: 'sequence', 2: 'vocab'}
        }
    )
    print("✅ Model exported to ONNX format successfully!")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    val_loss = checkpoint['best_val_loss'].item()
    print(f"val loss: {val_loss}")

    # robust runtime fetch
    runtime = _get_runtime_from_artifact(api, artifact)

    tokenizer = {
        'vocab_size': vocab_size,
        'vocab': {'stoi': stoi, 'itos': itos},
        'num_params': num_params,
        'val_loss': val_loss,
        'runtime_seconds': runtime
    }

    with open('output/tokenizer.json', 'w') as f:
        json.dump(tokenizer, f, indent=2)

    print("✅ Tokenizer configuration saved to tokenizer.json")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python convert.py <artifact_id>")
        sys.exit(1)

    artifact_id = sys.argv[1]
    artifact_path = f'jorgeeduardodsc/mini-models/model-{artifact_id}:v0'

    wandb.login()
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    print(f"Downloaded artifact: {artifact_path}")
    export_to_onnx(artifact_dir, artifact, api)
