# Audio V5.4 Decoder Architecture

## Purpose
- Document the current `v5.4` speech decoder stack in a concrete way.
- Keep the focus on the decoder side first:
  - coarse structure
  - fine detail
  - waveform reconstruction

## High-Level Inference Path
```text
prompt audio
  |
  +--> semantic tokenizer --------------------+
  |                                           |
  +--> EnCodec prompt codec frames ---------- | --------------------+
  |                                           |                     |
  +--> prompt prosody ----------------------- | ------------------+ |
                                              v                  | |
                                   semantic AR model            | |
                                   predicts next semantic       | |
                                   tokens                        | |
                                              |                  | |
                                              v                  | |
                               semantic frame upsampler          | |
                               (~25 Hz -> codec frame rate)      | |
                                              |                  | |
                                              v                  | |
                         coarse decoder (AR, codebook 0 only)    | |
                         predicts broad speech structure         | |
                                              |                  | |
                                              +------------------+ |
                                                                   v
                             fine decoder (non-AR, codebooks 1..7)
                             predicts residual/detail codec streams
                                                                   |
                                                                   v
                                                          EnCodec decode
                                                                   |
                                                                   v
                                                               waveform
```

## Mental Model
```text
semantic model = meaning / content
coarse decoder = speech skeleton
fine decoder = clarity / detail
EnCodec decode = waveform renderer
```

## Training-Time View
```text
1. semantic model
   input:
   - semantic prompt tokens
   target:
   - next semantic tokens

2. coarse decoder
   inputs:
   - semantic tokens
   - prompt codec prefix
   - prompt prosody
   target:
   - codec codebook 0 for future frames

3. fine decoder
   inputs:
   - semantic tokens
   - coarse codebook 0 sequence
   - prompt codec prefix
   - prompt prosody
   target:
   - codec codebooks 1..7 for future frames
```

## Why The Split Helps
- The old decoder tried to jump from semantic tokens directly to all codec codebooks.
- That was too much in one step:
  - content
  - timing
  - coarse acoustics
  - fine acoustics
- `v5.4` separates those jobs:
  - AR for coarse speech structure
  - non-AR detail prediction for residual codebooks

## Decoder Components

### 1. Semantic Frame Upsampler
File: [audio_semantic_upsampler.py](/home/jd/projects/aiplayground/nanoGPT/audio_semantic_upsampler.py)

- Input: low-rate semantic embeddings
- Output: frame-rate semantic features
- Mechanism:
  - repeat-interleave by `semantic_to_codec_ratio`
  - small Conv1d refinement stack

Shape idea:
```text
[B, semantic_steps, C] -> [B, codec_frames, C]
```

### 2. Coarse Decoder
File: [audio_coarse_decoder.py](/home/jd/projects/aiplayground/nanoGPT/audio_coarse_decoder.py)

- Predicts only codec codebook `0`
- Causal/autoregressive over frames
- Conditions on:
  - upsampled semantic memory
  - prompt coarse codec prefix
  - prompt prosody

Internals:
- semantic embedding + upsampling
- prompt embedding + prosody projection
- transformer encoder for conditioning memory
- transformer decoder for AR coarse prediction

Shape idea:
```text
semantic tokens + prompt info -> memory
previous coarse tokens -> causal decoder
decoder hidden -> logits over codec vocab
```

### 3. Fine Decoder
File: [audio_fine_decoder.py](/home/jd/projects/aiplayground/nanoGPT/audio_fine_decoder.py)

- Predicts residual codebooks `1..7`
- Not autoregressive over residual codebooks
- Conditions on:
  - upsampled semantic memory
  - coarse codebook sequence
  - prompt codec prefix
  - prompt prosody

Internals:
- semantic embedding + upsampling
- prompt codec/prosody memory encoder
- coarse-token target-side input
- transformer decoder
- one output head per residual codebook

Shape idea:
```text
semantic tokens + coarse tokens + prompt info
-> hidden states per frame
-> 7 residual heads
-> residual codebooks 1..7
```

## End-To-End Sampling Files
- semantic + old one-stage path:
  - [sample_audio_v5.py](/home/jd/projects/aiplayground/nanoGPT/sample_audio_v5.py)
- semantic + coarse-only diagnostic path:
  - [sample_audio_v5_coarse.py](/home/jd/projects/aiplayground/nanoGPT/sample_audio_v5_coarse.py)
- semantic + coarse + fine two-stage path:
  - [sample_audio_v5_two_stage.py](/home/jd/projects/aiplayground/nanoGPT/sample_audio_v5_two_stage.py)

## Current Diagnostic Reading
- If `ground_truth semantic + ground_truth coarse` still sounds bad:
  - fine decoder is the immediate bottleneck
- If `ground_truth/ground_truth` sounds much better than `predicted/predicted`:
  - semantic and/or coarse stages are the next bottlenecks

## Current Priority
- Train the coarse decoder well.
- Train the fine decoder well.
- Judge decoder quality first.
- Only after decoder quality is strong enough, spend effort improving semantic representations further.
