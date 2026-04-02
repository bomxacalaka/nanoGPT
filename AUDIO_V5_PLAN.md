# Audio V5 Plan

## Goal
- Keep an autoregressive speech model, but move the autoregressive burden onto lower-rate semantic units instead of dense acoustic tokens.
- Preserve direct audio-in and audio-out generation while making training and sampling faster than the current flattened-token and `v4` paths.
- Improve semantic behavior on tasks like spoken arithmetic by giving the model a cleaner abstraction level for "meaning" than raw codec tokens.

## Core Idea
- Split speech representation into two levels:
  - semantic tokens for content and long-range structure
  - acoustic tokens for speaker identity, prosody, and waveform detail
- Train the main autoregressive transformer only on semantic tokens.
- Use a second conditioned model to predict acoustic tokens from semantic tokens plus the prompt audio style.
- Decode acoustic tokens back to waveform with a pretrained neural codec.

## V5 Pipeline
1. Input waveform
2. Pretrained semantic tokenizer
3. Pretrained acoustic codec
4. Dataset stores aligned semantic and acoustic streams
5. Semantic AR model predicts future semantic units
6. Acoustic decoder predicts future codec frames conditioned on predicted semantics and prompt acoustic context
7. Pretrained codec decoder reconstructs waveform

## Why This Should Beat V4
- `v4` still autoregresses every coarse frame, which is much cheaper than `v2` but still tied to acoustic timing.
- `v5` moves AR to an even slower and more semantic stream.
- This cuts decode steps, improves long-range planning, and gives a better target for spoken reasoning tasks.

## Recommended Tokenizers
- Semantic stream:
  - preferred: HuBERT-style clustered units or SpeechTokenizer semantic tokens
  - fallback: coarse EnCodec stream grouped to a lower rate
- Acoustic stream:
  - keep the current pretrained `encodec_24khz` path first

## Training Stages

### Stage 1: Semantic Pretraining
- Train an autoregressive transformer over semantic tokens only.
- Dataset:
  - speech-only corpora first
  - Mini LibriSpeech is enough for smoke tests, but larger speech is needed for real gains
- Objective:
  - next semantic token prediction

### Stage 2: Acoustic Reconstruction
- Train a non-AR or lightly AR decoder that predicts acoustic codec frames from:
  - semantic tokens
  - prompt acoustic frames from the prefix
- Objective:
  - cross-entropy on codec codebooks

### Stage 3: Math Fine-Tuning
- Fine-tune the semantic AR model on synthetic spoken arithmetic.
- Fine-tune the acoustic decoder afterward on the same data.
- Keep evaluation on:
  - teacher-forced val loss
  - ASR transcript correctness
  - arithmetic correctness after ASR

## Repo Changes

### New Files
- `audio_semantic_tokenizer.py`
  - wrapper around the semantic tokenizer backend
  - converts waveform into semantic token ids
- `audio_semantic_model.py`
  - semantic AR transformer
- `audio_acoustic_decoder.py`
  - conditioned acoustic decoder from semantic tokens to codec frames
- `data/audio_semantic_codec/prepare.py`
  - builds aligned semantic/acoustic training shards
- `config/train_audio_semantic_v5.py`
  - base speech semantic AR config
- `config/train_audio_semantic_math_v5.py`
  - arithmetic semantic fine-tune config
- `config/train_audio_acoustic_decoder_v5.py`
  - acoustic decoder config
- `sample_audio_v5.py`
  - semantic generate -> acoustic decode -> waveform

### Existing Files to Extend
- `train.py`
  - add model-type support for:
    - semantic AR model
    - acoustic decoder model
- `sample_audio.py`
  - either dispatch to `v5` or leave `v5` isolated in `sample_audio_v5.py`
- `audio_codec.py`
  - keep as the acoustic codec wrapper

## Dataset Format
- New dataset directory:
  - `data/audio_semantic_codec/`
- Files:
  - `train.semantic.bin`
  - `val.semantic.bin`
  - `train.codec.bin`
  - `val.codec.bin`
  - `train.align.npy`
  - `val.align.npy`
  - `meta.pkl`
- `meta.pkl` should store:
  - semantic tokenizer type and rate
  - codec model and bandwidth
  - codebook count and codebook size
  - alignment strategy between semantic steps and codec frames

## Alignment Rule
- Keep the first version simple:
  - each semantic token spans a fixed number of codec frames
- If the tokenizer emits variable-rate units later, add an explicit alignment table per utterance.

## Sampling Path
1. Encode prompt audio into semantic tokens and codec prompt frames
2. Run semantic AR continuation
3. Condition acoustic decoder on:
  - prompt codec frames
  - prompt semantic tokens
  - generated semantic continuation
4. Decode full codec sequence to waveform

## Metrics
- Semantic AR:
  - train loss
  - val loss
  - semantic token perplexity
- End-to-end speech:
  - codec reconstruction loss
  - generated seconds per second of compute
  - ASR word error rate
  - arithmetic accuracy after ASR for math data

## Immediate Implementation Order
1. Add `audio_semantic_tokenizer.py` with a minimal semantic backend interface
2. Add `data/audio_semantic_codec/prepare.py` using:
  - current EnCodec for acoustic tokens
  - placeholder semantic backend that can later be swapped
3. Add `audio_semantic_model.py`
4. Train semantic-only `v5` on Mini LibriSpeech
5. Add `audio_acoustic_decoder.py`
6. Train end-to-end `v5` reconstruction/generation
7. Fine-tune on spoken arithmetic

## Practical First Cut
- For the first `v5` milestone, do not chase perfect tokenizer research integration.
- Use a pluggable semantic backend interface, even if the first backend is crude.
- The important thing is the architecture split:
  - semantic AR model
  - acoustic conditioned decoder

## Target Outcome
- Faster generation than `v2`
- Better semantic planning than `v4`
- Cleaner path toward spoken reasoning and controllable speech generation
