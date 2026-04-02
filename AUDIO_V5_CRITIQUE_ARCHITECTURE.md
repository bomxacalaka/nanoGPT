# Audio V5.5 Critique Architecture

## Purpose
- Define the next step after `v5.4` decoder work.
- Keep the generator architecture intact:
  - semantic model
  - coarse decoder
  - fine decoder
- Add a structured critique layer that says what is wrong, not just `real` or `fake`.

## Core Idea
Instead of a plain GAN discriminator:
- one network generates speech
- another network critiques specific failure modes

That critique should answer:
- is the audio realistic?
- is the speech intelligible?
- did the continuation keep the prompt speaking style?
- did the continuation keep the prompt prosody?
- did the continuation preserve the intended semantic content?

## High-Level Diagram
```text
prompt audio
  |
  +--> generator stack (v5.4)
  |     - semantic model
  |     - coarse decoder
  |     - fine decoder
  |     - EnCodec decode
  |
  +--> generated continuation audio -------------------+
  |                                                    |
  +--> target continuation audio --------------------+ |
  |                                                  | |
  +--> prompt audio -------------------------------+ | |
                                                   v v v
                                         critique stack
                                           - realism head
                                           - intelligibility head
                                           - style head
                                           - prosody head
                                           - semantic head
```

## Why Not Plain GAN First
- A binary `real/fake` signal is too weak for this project.
- It can push realism without improving words.
- It is harder to stabilize.

The better path is:
1. structured critique metrics first
2. supervised + auxiliary losses
3. only later, optional low-weight adversarial realism loss

## Critique Heads

### 1. Realism Head
Question:
- Does this continuation sound like valid human speech audio at all?

Target:
- real continuation vs generated continuation

Use:
- polish audio naturalness
- later adversarial realism if needed

### 2. Intelligibility Head
Question:
- Are phonetic units and words recoverable clearly?

Target:
- compare generated continuation against target continuation

Use:
- punish blurry, speech-like-but-unreadable output

### 3. Style Head
Question:
- Does the continuation still sound like the same speaker/style as the prompt?

Target:
- prompt audio vs generated continuation
- prompt audio vs target continuation

Use:
- preserve vocal identity / timbre / delivery style

### 4. Prosody Head
Question:
- Did rhythm, energy, voicing, and speaking feel remain consistent?

Target:
- generated prosody vs target prosody
- optionally prompt prosody similarity for short continuation tasks

Use:
- preserve timing, pacing, emphasis, and general speaking behavior

### 5. Semantic Head
Question:
- Did the continuation preserve the intended content?

Target:
- generated semantic units vs target semantic units
- or generated audio passed back through semantic tokenizer

Use:
- punish semantic drift even when audio sounds natural

## Practical First Version
Do not start with a learned adversarial critic.
Start with structured critique metrics that can be logged and later turned into losses.

### Non-Adversarial Critique Metrics
1. Semantic consistency
- Re-encode generated audio with the semantic tokenizer
- Compare against target semantic tokens

2. Prosody consistency
- Extract pitch / energy / voiced features from generated continuation
- Compare against target continuation prosody

3. Coarse-code consistency
- Re-encode generated audio with EnCodec
- Compare codebook `0` against target codebook `0`

4. Residual-code consistency
- Re-encode generated audio with EnCodec
- Compare residual codebooks `1..7` against target residual codebooks

These are critique signals even if they are not yet fully differentiable through the whole stack.

## Proposed Learned Critic
File to add later:
- [audio_critic.py](/home/jd/projects/aiplayground/nanoGPT/audio_critic.py)

Suggested inputs:
- prompt mel spectrogram
- generated continuation mel spectrogram
- target continuation mel spectrogram

Suggested outputs:
- `realism_score`
- `intelligibility_score`
- `style_match_score`
- `prosody_match_score`
- `semantic_match_score`

## Suggested Loss Stack
When training starts to use the critique signals, the generator objective should look more like:

```text
L_total =
  L_coarse_ce
+ L_fine_ce
+ L_semantic_consistency
+ L_prosody_consistency
+ L_style_consistency
+ small_weight * L_realism_adv
```

Important:
- do not start with the adversarial term
- make realism a small auxiliary pressure, not the main objective

## Recommended Implementation Order
1. Add critique reporting to sampling/evaluation:
- semantic consistency
- prosody consistency
- coarse-code consistency
- residual-code consistency

2. Add a standalone learned critic model:
- no adversarial training yet
- just train it to score target vs generated quality dimensions

3. Add generator-side auxiliary training signals where practical.

4. Add low-weight realism adversarial loss only after the above works.

## Repo Direction
Current order of work should be:
1. make `v5.4` decoder strong
2. add structured critique metrics
3. add `v5.5` learned critique model
4. only then consider adversarial training

## Main Principle
The critique model should act like a speech report card, not a yes/no gate.

Good critique output should look conceptually like:
```text
realism: weak
intelligibility: poor
style match: medium
prosody match: good
semantic match: poor
```

That is far more useful for this project than:
```text
fake
```
