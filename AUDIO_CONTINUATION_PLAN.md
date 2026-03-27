# Audio Continuation Plan

## Goal
- Train an autoregressive transformer that continues audio from a real audio prompt.
- Keep the learning setup as close as possible to current `nanoGPT` training:
  - next-token prediction
  - causal transformer
  - cross-entropy loss
- Avoid raw waveform regression for the first version.

## Chosen Approach
- Use a pretrained neural audio codec to convert waveform audio into discrete tokens.
- Train the transformer on codec-token sequences exactly like text tokens.
- For inference:
  - encode a prompt audio clip into codec tokens
  - ask the model to predict more tokens
  - decode the full token sequence back into waveform audio

## Why This Approach
- It matches how this repo already works:
  - integer token sequences
  - contiguous chunk sampling
  - next-token prediction
- It avoids a larger rewrite into continuous spectrogram regression.
- It is the most direct path to a "GPT for audio continuation" prototype.

## Recommended Stack
- Codec: `EnCodec`
- PyTorch remains the training backend
- Keep the existing transformer training loop as much as possible

## First Version Constraints
- Use one audio domain only:
  - either speech
  - or music
  - or one narrow sound class
- Use mono audio first
- Use one fixed sample rate first
- Use short clips first
- Use flattened codec streams first, even if it is not the final ideal architecture

## Token Representation
- EnCodec commonly returns multiple codebooks per frame.
- First implementation should flatten them into one long token stream.
- Example flattening idea:
  - `frame0_book0`
  - `frame0_book1`
  - `frame0_book2`
  - ...
  - `frame1_book0`
  - `frame1_book1`
  - ...
- Each codebook should be offset into a shared global vocabulary so token IDs remain unique.
- Store the flattening scheme in `meta.pkl`.

## Repo Changes

### 1. Add audio dataset prep
- New folder: `data/audio_codec/`
- New file: [`data/audio_codec/prepare.py`](/home/jd/projects/aiplayground/nanoGPT/data/audio_codec/prepare.py)
- Responsibilities:
  - discover audio files
  - load waveform audio
  - resample to one sample rate
  - convert stereo to mono if needed
  - chunk long files into fixed or bounded windows
  - encode each chunk with EnCodec
  - flatten codec tokens into one integer stream
  - split train / val
  - write `train.bin`, `val.bin`, `meta.pkl`

### 2. Add dataset metadata
- `meta.pkl` should include:
  - `dataset_type = "audio_codec"`
  - codec name and version
  - sample rate
  - codebook count
  - codebook size
  - vocab size
  - flattening order
  - optional file manifest summary

### 3. Add training config
- New file: [`config/train_audio_codec.py`](/home/jd/projects/aiplayground/nanoGPT/config/train_audio_codec.py)
- First version should stay modest:
  - shorter block size
  - smaller model
  - smaller batch size
  - no distributed complexity until the pipeline works end to end

### 4. Keep `train.py` mostly unchanged
- [`train.py`](/home/jd/projects/aiplayground/nanoGPT/train.py) already assumes integer tokens in `train.bin` and `val.bin`.
- Expected edits should be small:
  - allow larger integer dtype if vocab exceeds `uint16`
  - improve metadata loading
  - possibly improve `sample.py` compatibility for audio token generation

### 5. Add audio sampling script
- New file: [`sample_audio.py`](/home/jd/projects/aiplayground/nanoGPT/sample_audio.py)
- Responsibilities:
  - load the model checkpoint
  - load codec + dataset metadata
  - encode a prompt waveform into tokens
  - run autoregressive continuation
  - unflatten tokens back into codec streams
  - decode tokens into waveform
  - save output `.wav`

### 6. Add audio utility module
- New file: [`audio_codec.py`](/home/jd/projects/aiplayground/nanoGPT/audio_codec.py)
- Responsibilities:
  - wrap codec loading
  - waveform preprocessing
  - token flattening / unflattening
  - prompt encode / generated decode helpers

## Minimal Milestones

### Milestone 1: Tokenization only
- Pick a small audio folder
- Encode a few files with EnCodec
- Verify:
  - tokens are produced
  - tokens can be decoded back to recognizable audio

### Milestone 2: Dataset build
- Create `train.bin`, `val.bin`, `meta.pkl`
- Verify:
  - token counts look sane
  - vocab size is correct
  - train / val split works

### Milestone 3: Tiny training run
- Train a very small model for a short run
- Verify:
  - loss decreases
  - generation code runs without shape or dtype errors

### Milestone 4: Prompted continuation
- Encode a real prompt clip
- Generate continuation tokens
- Decode to audio
- Judge quality by listening first

### Milestone 5: Improve quality
- Tune:
  - block size
  - model size
  - dataset domain cleanliness
  - flattening strategy
  - generation temperature / top-k

## Known Risks
- Codec token streams can still be very long.
- Flattening multiple codebooks is simple but not optimal.
- Mixed domains will likely produce poor continuation quality.
- Longer generations will degrade before short ones do.
- Audio quality is bounded by codec quality and dataset consistency.

## Non-Goals For Version 1
- Raw waveform autoregression
- Mel-spectrogram regression
- CNN frontends
- Multi-stage hierarchical audio models
- Text conditioning
- Real-time generation

## Immediate Next Implementation Step
- Build `data/audio_codec/prepare.py` first.
- The first proof point is:
  - load a `.wav`
  - encode it with EnCodec
  - save a flattened token stream
  - decode it back successfully
