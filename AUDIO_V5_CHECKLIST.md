# Audio V5 Checklist

## Goal
- Build a new speech-native model from scratch.
- Input is audio, output is audio.
- The model should preserve and understand meaning plus expressive cues:
  - emotion
  - tone
  - whispering
  - breathiness
  - hesitation
  - non-speech sounds
- Keep the core reasoning model autoregressive, but run AR on semantic speech units instead of dense acoustic tokens.

## Architecture
- Frontend:
  - semantic speech tokenizer
  - expressive/style feature extractor
  - acoustic codec tokenizer
- Core model:
  - autoregressive transformer over semantic tokens
- Decoder:
  - conditioned acoustic generator from semantics plus prompt style/context
- Output:
  - codec decoder back to waveform

## Paper Pattern
- AudioLM:
  - use separate semantic and acoustic token levels
- SpeechTokenizer:
  - disentangle speech content from acoustic detail
- Spirit LM:
  - keep open the path to deeper speech-level meaning modeling
- SoundStorm:
  - do not make acoustic detail generation fully AR if speed matters

## Step By Step

### Phase 1: Spec
- Lock the `v5` contract:
  - audio in
  - semantic AR middle
  - audio out
- Keep this as a brand-new path, not a patch on older `v2`/`v3`/`v4`.

### Phase 2: Semantic Tokenizer
- Create `audio_semantic_tokenizer.py`
- Support a pluggable semantic backend
- First version requirements:
  - waveform -> semantic token ids
  - save/load tokenizer metadata
  - deterministic offline preprocessing

### Phase 3: Acoustic Codec
- Keep a pretrained codec backend for waveform reconstruction
- Reuse the existing EnCodec wrapper first
- Requirements:
  - waveform -> acoustic codec tokens
  - codec tokens -> waveform

### Phase 4: Dataset Format
- Create `data/audio_semantic_codec/prepare.py`
- For each utterance, store:
  - semantic tokens
  - acoustic codec tokens
  - alignment info between the two streams
  - metadata for sample rate, tokenizer types, codebook counts, and rates
- Start with a fixed alignment rule

### Phase 5: Semantic AR Model
- Create `audio_semantic_model.py`
- Train a transformer only on semantic tokens
- First objective:
  - next semantic token prediction
- First dataset:
  - Mini LibriSpeech or equivalent speech corpus

### Phase 6: Acoustic Decoder
- Create `audio_acoustic_decoder.py`
- Condition on:
  - prompt acoustic prefix
  - prompt semantic tokens
  - generated semantic continuation
- Predict future acoustic codec frames
- Prefer parallel or lightly autoregressive decoding

### Phase 7: End-To-End Sampling
- Create `sample_audio_v5.py`
- Pipeline:
  - encode prompt audio
  - generate semantic continuation
  - generate acoustic continuation
  - decode waveform
- Save:
  - prompt
  - full output
  - continuation-only output

### Phase 8: Base Speech Training
- Train the semantic AR model on speech data
- Train the acoustic decoder after that
- Verify:
  - semantic val loss
  - generation speed
  - intelligibility
  - speaker/style carryover from prompt

### Phase 9: Spoken Math Fine-Tuning
- Build a larger synthetic spoken-math corpus
- Keep multiple voices and varied speaking styles
- Fine-tune:
  - semantic AR model
  - then acoustic decoder
- Evaluate:
  - ASR transcript accuracy
  - arithmetic correctness
  - spoken naturalness

### Phase 10: Expressive Speech Features
- Add explicit style/prosody conditioning
- Candidate features:
  - pitch
  - energy
  - speaking rate
  - pause timing
  - style embedding from prompt audio
- Goal:
  - respond with meaning plus expressive control

### Phase 11: Better Supervision
- Add speech understanding evaluation, not just token loss
- Candidate checks:
  - ASR WER
  - spoken math correctness
  - emotion/style preservation
- Later:
  - preference optimization or semantic reward tuning

### Phase 12: Scaling
- Increase corpus size once the first end-to-end path works
- Keep semantic AR as the main reasoning bottleneck
- Keep acoustic generation efficient

## Non-Goals For First V5 Milestone
- Do not try to solve every emotion/style/control problem immediately
- Do not keep stretching the old flattened-token setup
- Do not optimize browser tooling or dashboards

## First Concrete Repo Tasks
1. Add `audio_semantic_tokenizer.py`
2. Add `data/audio_semantic_codec/prepare.py`
3. Add `audio_semantic_model.py`
4. Add `audio_acoustic_decoder.py`
5. Add `sample_audio_v5.py`
6. Extend `train.py` for the new model types
7. Add base `v5` configs
8. Run the first speech-only smoke test
