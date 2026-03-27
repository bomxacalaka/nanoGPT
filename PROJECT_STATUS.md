# Project Status

## Purpose
- Base repo is `nanoGPT`, adapted into a small custom character-level GPT playground.
- Current visible direction is a tiny arithmetic language model plus a browser inference/export path.
- Secondary direction exists for text-to-JSON synthetic data generation, but that does not appear to be the main active demo right now.

## Current Model
- Training entrypoint is still [`train.py`](/home/jd/projects/aiplayground/nanoGPT/train.py) with the standard GPT-style decoder-only architecture from [`model.py`](/home/jd/projects/aiplayground/nanoGPT/model.py).
- Main active config appears to be [`config/train_shakespeare_char.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_shakespeare_char.py).
- Current config values:
  - Dataset: `shakespeare_char`
  - Output dir: `out-shakespeare-char`
  - Context length: `128`
  - Layers / heads / embedding: `6 / 6 / 120`
  - Dropout: `0.3`
  - Batch size: `64`
  - Gradient accumulation: `8`
  - Max iters: `100000`
  - W&B: enabled, project `mini-models`, run name base `vibe_maths`
- Latest checkpoint currently in [`out-shakespeare-char/ckpt.pt`](/home/jd/projects/aiplayground/nanoGPT/out-shakespeare-char/ckpt.pt) reports:
  - `iter_num`: `1000`
  - `best_val_loss`: `0.1658516824245453`
  - Model args: `n_layer=6`, `n_head=6`, `n_embd=120`, `block_size=128`, `bias=False`, `vocab_size=28`, `dropout=0.3`

## Tokenization
- Tokenizer is custom and character-level, defined in [`tokenizer.py`](/home/jd/projects/aiplayground/nanoGPT/tokenizer.py).
- Special-token support exists in code, but is currently disabled by default with `special_tokens = []`.
- Current exported vocabulary size is `28`.
- Current exported vocabulary is mostly lowercase letters plus newline, space, `-`, `<`, `>`.
- This means the arithmetic model is learning text patterns at the character level, not number/operator tokens.

## Training Data
- Active prepared dataset metadata in [`data/shakespeare_char/meta.pkl`](/home/jd/projects/aiplayground/nanoGPT/data/shakespeare_char/meta.pkl) points to [`data/shakespeare_char/000_999.txt`](/home/jd/projects/aiplayground/nanoGPT/data/shakespeare_char/000_999.txt).
- That file contains arithmetic statements written in words, e.g. `<zero plus zero equals zero>`.
- Current prepared split in [`data/shakespeare_char/prepare.py`](/home/jd/projects/aiplayground/nanoGPT/data/shakespeare_char/prepare.py):
  - Uses a `99% / 1%` train/val split
  - Splits on `>\n` to avoid cutting through a record
  - Writes `train.bin`, `val.bin`, and `meta.pkl`
- Other dataset files in the same folder suggest prior experiments with larger or alternate corpora:
  - `0_9.txt`
  - `0_999.txt`
  - `base.txt`
  - `corp.txt`
  - `disease.txt`
  - `input_old.txt`
  - `input_test.txt`

## Export / Inference
- There is an active ONNX/browser path in [`convert/convert.py`](/home/jd/projects/aiplayground/nanoGPT/convert/convert.py) and [`website/infer.html`](/home/jd/projects/aiplayground/nanoGPT/website/infer.html).
- Browser demo branding currently says `Vibe Maths`.
- Exported browser assets already exist in [`out-shakespeare-char/ckpt.onnx`](/home/jd/projects/aiplayground/nanoGPT/out-shakespeare-char/ckpt.onnx) and [`out-shakespeare-char/tokenizer.json`](/home/jd/projects/aiplayground/nanoGPT/out-shakespeare-char/tokenizer.json).
- The website currently loads `model.onnx` and `tokenizer.json`, so path alignment may need checking before deployment.

## Results And Evidence
- Best explicit metric found so far is `best_val_loss ~= 0.166` from the latest checkpoint.
- [`evolution.txt`](/home/jd/projects/aiplayground/nanoGPT/evolution.txt) looks like a raw generations scratchpad showing failure modes and malformed arithmetic outputs from earlier runs.
- Existing output folders suggest multiple training/export attempts:
  - `out-shakespeare-char`
  - `out-shakespeare-char-best`
  - `out-shakespeare-char-json-base`
  - `out-shakespeare-char_old`
  - `out-shakespeare`
  - `out-shakespeare_old`

## History So Far
- Started from upstream `nanoGPT`.
- Added a custom character tokenizer and custom dataset preparation flow.
- Shifted at least one main experiment toward arithmetic-in-words generation.
- Added synthetic data generation for a separate text-to-JSON task in [`data/synthetic_gen/gen.py`](/home/jd/projects/aiplayground/nanoGPT/data/synthetic_gen/gen.py), which writes training-style records into [`data/shakespeare/input.txt`](/home/jd/projects/aiplayground/nanoGPT/data/shakespeare/input.txt).
- Added ONNX export plus a lightweight browser inference UI.
- Repo currently has signs of in-progress work:
  - Deleted tracked file: `export_onnx.py`
  - Modified tracked file: `sample.ipynb`
  - Untracked directories: `convert/`, `website/`

## Working Notes
- Keep this file short and summary-first.
- Update bullets in place instead of appending long chronological logs.
- If a change is temporary, note it under `History So Far` or replace stale bullets instead of expanding the file indefinitely.
