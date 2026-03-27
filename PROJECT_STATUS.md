# Project Status

## Purpose
- Base repo is `nanoGPT`, adapted into a small custom character-level GPT playground.
- Current visible directions are:
  - a tiny arithmetic language model plus a browser inference/export path
  - a newer audio-continuation branch using discrete EnCodec token streams
- Secondary direction exists for text-to-JSON synthetic data generation, but that does not appear to be the main active demo right now.

## Current Model
- Training entrypoint is still [`train.py`](/home/jd/projects/aiplayground/nanoGPT/train.py) with the standard GPT-style decoder-only architecture from [`model.py`](/home/jd/projects/aiplayground/nanoGPT/model.py).
- Text config:
  - [`config/train_shakespeare_char.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_shakespeare_char.py)
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
- Active audio config:
  - [`config/train_audio_librispeech_v2.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_audio_librispeech_v2.py)
  - Dataset: `audio_librispeech_codec`
  - Output dir: `out-audio-librispeech-v2`
  - Context length: `2048`
  - Layers / heads / embedding: `8 / 8 / 256`
  - Dropout: `0.1`
  - Batch size: `8`
  - Gradient accumulation: `4`
  - Max iters: `20000`

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
- Active audio dataset is [`data/audio_librispeech_codec/`](/home/jd/projects/aiplayground/nanoGPT/data/audio_librispeech_codec/), prepared from:
  - train split: `Mini LibriSpeech train-clean-5`
  - validation split: `Mini LibriSpeech dev-clean-2`
  - raw size: about `902M`
  - source files: `1519` train, `1089` val
  - token counts: `11,537,988` train, `4,423,926` val
  - codec setup: `encodec_24khz`, bandwidth `6.0`, `8` codebooks, `600` flattened tokens/sec, vocab size `8193`

## Export / Inference
- There is an active ONNX/browser path in [`convert/convert.py`](/home/jd/projects/aiplayground/nanoGPT/convert/convert.py) and [`website/infer.html`](/home/jd/projects/aiplayground/nanoGPT/website/infer.html).
- Browser demo branding currently says `Vibe Maths`.
- Exported browser assets already exist in [`out-shakespeare-char/ckpt.onnx`](/home/jd/projects/aiplayground/nanoGPT/out-shakespeare-char/ckpt.onnx) and [`out-shakespeare-char/tokenizer.json`](/home/jd/projects/aiplayground/nanoGPT/out-shakespeare-char/tokenizer.json).
- The website currently loads `model.onnx` and `tokenizer.json`, so path alignment may need checking before deployment.

## Audio Work
- Audio continuation is working around discrete EnCodec token streams rather than mel-spectrogram regression.
- New audio files:
  - [`audio_codec.py`](/home/jd/projects/aiplayground/nanoGPT/audio_codec.py)
  - [`data/audio_codec/prepare.py`](/home/jd/projects/aiplayground/nanoGPT/data/audio_codec/prepare.py)
  - [`config/train_audio_codec.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_audio_codec.py)
  - [`config/train_audio_librispeech.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_audio_librispeech.py)
  - [`config/train_audio_librispeech_v2.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_audio_librispeech_v2.py)
  - [`AUDIO_CONTINUATION_PLAN.md`](/home/jd/projects/aiplayground/nanoGPT/AUDIO_CONTINUATION_PLAN.md)
  - [`sample_audio.py`](/home/jd/projects/aiplayground/nanoGPT/sample_audio.py)
  - [`sample_audio_from_text.py`](/home/jd/projects/aiplayground/nanoGPT/sample_audio_from_text.py)
- Current audio path status:
  - `encodec` installed in the `aiplayground` conda environment
  - MeloTTS prompt testing now works from text by loading the official repo from `/tmp/MeloTTS`
  - extra runtime packages added in `aiplayground` to satisfy MeloTTS under Python 3.12, including `fugashi` and `soxr`
  - helper code verified to encode and decode synthetic and real 24 kHz mono clips
  - flattened-codebook sampling is constrained by valid codebook ranges, so generated token streams stay decodable
  - first smoke dataset was Free Spoken Digit Dataset (`3000` clips, about `42M` raw)
  - current phrase-level dataset is Mini LibriSpeech in [`data/audio_librispeech_codec/`](/home/jd/projects/aiplayground/nanoGPT/data/audio_librispeech_codec/)
  - current best phrase-level checkpoint in [`out-audio-librispeech-v2/ckpt.pt`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/ckpt.pt) reports:
    - `iter_num`: `250`
    - `best_val_loss`: `5.5064`
    - model args: `n_layer=8`, `n_head=8`, `n_embd=256`, `block_size=2048`, `vocab_size=8193`, `dropout=0.1`
  - a smaller prior LibriSpeech baseline still exists in [`out-audio-librispeech/ckpt.pt`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/ckpt.pt) with `best_val_loss ~= 5.838`
  - end-to-end prompt-to-waveform sampling succeeded with [`sample_audio.py`](/home/jd/projects/aiplayground/nanoGPT/sample_audio.py)
  - spoken-digit example generated file: [`out-audio-codec/generated_3_nicolas_7.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-codec/generated_3_nicolas_7.wav)
  - training-time audio progress now saves prompt, full output, and continuation-only clips under [`out-audio-librispeech-v2/samples/`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples)
  - sampling scripts now warn when prompt duration exceeds the model's effective context window

## Results And Evidence
- Best explicit metric found so far is `best_val_loss ~= 0.166` from the latest checkpoint.
- Best phrase-level audio run so far is `best_val_loss ~= 5.506` on Mini LibriSpeech with the wider-context `v2` config.
- Saved `v2` continuation-only samples currently include:
  - [`out-audio-librispeech-v2/samples/step_000000_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000000_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000050_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000050_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000100_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000100_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000150_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000150_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000200_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000200_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000250_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000250_continuation.wav)
  - [`out-audio-librispeech-v2/samples/step_000300_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech-v2/samples/step_000300_continuation.wav)
- A text-prompt smoke test now exists under [`out-audio-librispeech/tts_prompt_tests/`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/tts_prompt_tests):
  - raw MeloTTS prompt: [`generated_from_text_prompt_trimmed.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/tts_prompt_tests/generated_from_text_prompt_trimmed.wav)
  - EnCodec round-trip prompt: [`generated_from_text_prompt_codec.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/tts_prompt_tests/generated_from_text_prompt_codec.wav)
  - full prompt-plus-continuation output: [`generated_from_text_full.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/tts_prompt_tests/generated_from_text_full.wav)
  - continuation-only clip: [`generated_from_text_continuation.wav`](/home/jd/projects/aiplayground/nanoGPT/out-audio-librispeech/tts_prompt_tests/generated_from_text_continuation.wav)
- [`evolution.txt`](/home/jd/projects/aiplayground/nanoGPT/evolution.txt) looks like a raw generations scratchpad showing failure modes and malformed arithmetic outputs from earlier runs.
- Existing output folders suggest multiple training/export attempts:
  - `out-audio-codec`
  - `out-audio-librispeech`
  - `out-audio-librispeech-v2`
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
- Started an EnCodec-based audio continuation path that preserves the repo's next-token training pattern.
- Pre-audio rollback checkpoint commit is `0e80c0d` (`checkpoint before audio continuation work`).
- Patched `train.py` so non-text token datasets can train without `stoi` / `itos` metadata and without the text-only arithmetic sampling hook.
- Repo currently has signs of in-progress work:
  - Deleted tracked file: `export_onnx.py`
  - Modified tracked file: `sample.ipynb`
  - Untracked directories: `convert/`, `website/`

## Working Notes
- Keep this file short and summary-first.
- Update bullets in place instead of appending long chronological logs.
- If a change is temporary, note it under `History So Far` or replace stale bullets instead of expanding the file indefinitely.
