#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-aiplayground}"
INPUT_DIR="${INPUT_DIR:-data/audio_librispeech_codec/raw/LibriSpeech/train-clean-5}"
VAL_INPUT_DIR="${VAL_INPUT_DIR:-data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2}"
DATASET_DIR="${DATASET_DIR:-data/audio_semantic_codec_hubert}"
CENTROIDS_PATH="${CENTROIDS_PATH:-data/audio_semantic_codec/hubert_kmeans_256.npy}"
HUBERT_BUNDLE_NAME="${HUBERT_BUNDLE_NAME:-HUBERT_BASE}"
HUBERT_LAYER="${HUBERT_LAYER:--1}"
HUBERT_FRAME_STRIDE="${HUBERT_FRAME_STRIDE:-2}"
NUM_CLUSTERS="${NUM_CLUSTERS:-256}"
MAX_CLUSTER_FRAMES="${MAX_CLUSTER_FRAMES:-100000}"
KMEANS_ITERS="${KMEANS_ITERS:-40}"
ASSIGN_BATCH_SIZE="${ASSIGN_BATCH_SIZE:-4096}"
CENTROID_DEVICE="${CENTROID_DEVICE:-cuda}"
SEMANTIC_DEVICE="${SEMANTIC_DEVICE:-cuda}"
CODEC_DEVICE="${CODEC_DEVICE:-cpu}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
TRAIN_DTYPE="${TRAIN_DTYPE:-float16}"
TRAIN_COMPILE="${TRAIN_COMPILE:-False}"
SEMANTIC_OUT_DIR="${SEMANTIC_OUT_DIR:-out-audio-semantic-v5-hubert}"
ACOUSTIC_OUT_DIR="${ACOUSTIC_OUT_DIR:-out-audio-acoustic-decoder-v5-hubert}"
SEMANTIC_BLOCK_SIZE="${SEMANTIC_BLOCK_SIZE:-256}"
ACOUSTIC_BLOCK_SIZE="${ACOUSTIC_BLOCK_SIZE:-384}"
PROMPT_AUDIO="${PROMPT_AUDIO:-data/audio_librispeech_codec/raw/LibriSpeech/dev-clean-2/174/168635/174-168635-0002.flac}"
OUTPUT_WAV="${OUTPUT_WAV:-out-audio-v5-hubert/generated.wav}"
PROMPT_MAX_SECONDS="${PROMPT_MAX_SECONDS:-1.0}"
MAX_NEW_SECONDS="${MAX_NEW_SECONDS:-1.5}"
SEMANTIC_SOURCE="${SEMANTIC_SOURCE:-predicted}"
SEMANTIC_INIT_FROM="${SEMANTIC_INIT_FROM:-scratch}"
ACOUSTIC_INIT_FROM="${ACOUSTIC_INIT_FROM:-scratch}"
SEMANTIC_MAX_ITERS="${SEMANTIC_MAX_ITERS:-}"
SEMANTIC_LR_DECAY_ITERS="${SEMANTIC_LR_DECAY_ITERS:-}"
SEMANTIC_WARMUP_ITERS="${SEMANTIC_WARMUP_ITERS:-}"
SEMANTIC_EVAL_INTERVAL="${SEMANTIC_EVAL_INTERVAL:-}"
ACOUSTIC_MAX_ITERS="${ACOUSTIC_MAX_ITERS:-}"
ACOUSTIC_LR_DECAY_ITERS="${ACOUSTIC_LR_DECAY_ITERS:-}"
ACOUSTIC_WARMUP_ITERS="${ACOUSTIC_WARMUP_ITERS:-}"
ACOUSTIC_EVAL_INTERVAL="${ACOUSTIC_EVAL_INTERVAL:-}"

log_stage() {
  local msg="$1"
  printf '\n========== %s ==========\n' "$msg"
}

run_conda() {
  conda run --no-capture-output -n "$CONDA_ENV" "$@"
}

append_train_overrides() {
  local -n arr_ref=$1
  local max_iters="$2"
  local lr_decay_iters="$3"
  local warmup_iters="$4"
  local eval_interval="$5"
  if [[ -n "$max_iters" ]]; then
    arr_ref+=("--max_iters=$max_iters")
  fi
  if [[ -n "$lr_decay_iters" ]]; then
    arr_ref+=("--lr_decay_iters=$lr_decay_iters")
  fi
  if [[ -n "$warmup_iters" ]]; then
    arr_ref+=("--warmup_iters=$warmup_iters")
  fi
  if [[ -n "$eval_interval" ]]; then
    arr_ref+=("--eval_interval=$eval_interval")
  fi
}

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Missing INPUT_DIR: $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -d "$VAL_INPUT_DIR" ]]; then
  echo "Missing VAL_INPUT_DIR: $VAL_INPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_AUDIO" ]]; then
  echo "Missing PROMPT_AUDIO: $PROMPT_AUDIO" >&2
  exit 1
fi

mkdir -p "$(dirname "$CENTROIDS_PATH")" "$(dirname "$OUTPUT_WAV")" "$DATASET_DIR"
OUTPUT_ROOT="${OUTPUT_WAV%.*}"
OUTPUT_EXT="${OUTPUT_WAV##*.}"

log_stage "Build HuBERT Centroids"
echo "Input audio: $INPUT_DIR"
echo "Centroids output: $CENTROIDS_PATH"
run_conda python -u data/audio_semantic_codec/build_hubert_kmeans.py \
  --input_dir "$INPUT_DIR" \
  --output_path "$CENTROIDS_PATH" \
  --bundle_name "$HUBERT_BUNDLE_NAME" \
  --layer "$HUBERT_LAYER" \
  --frame_stride "$HUBERT_FRAME_STRIDE" \
  --num_clusters "$NUM_CLUSTERS" \
  --max_frames "$MAX_CLUSTER_FRAMES" \
  --kmeans_iters "$KMEANS_ITERS" \
  --assign_batch_size "$ASSIGN_BATCH_SIZE" \
  --device "$CENTROID_DEVICE"

log_stage "Prepare HuBERT v5 Dataset"
echo "Dataset output: $DATASET_DIR"
run_conda python -u data/audio_semantic_codec/prepare.py \
  --input_dir "$INPUT_DIR" \
  --val_input_dir "$VAL_INPUT_DIR" \
  --output_dir "$DATASET_DIR" \
  --semantic_backend hubert_kmeans_v1 \
  --hubert_centroids_path "$CENTROIDS_PATH" \
  --hubert_bundle_name "$HUBERT_BUNDLE_NAME" \
  --hubert_layer "$HUBERT_LAYER" \
  --hubert_frame_stride "$HUBERT_FRAME_STRIDE" \
  --semantic_device "$SEMANTIC_DEVICE" \
  --codec_device "$CODEC_DEVICE"

log_stage "Train Semantic Model"
semantic_cmd=(
  python -u train.py config/train_audio_semantic_v5.py
  "--dataset=$(basename "$DATASET_DIR")"
  "--out_dir=$SEMANTIC_OUT_DIR"
  "--device=$TRAIN_DEVICE"
  "--dtype=$TRAIN_DTYPE"
  "--compile=$TRAIN_COMPILE"
  "--init_from=$SEMANTIC_INIT_FROM"
  "--block_size=$SEMANTIC_BLOCK_SIZE"
)
append_train_overrides semantic_cmd "$SEMANTIC_MAX_ITERS" "$SEMANTIC_LR_DECAY_ITERS" "$SEMANTIC_WARMUP_ITERS" "$SEMANTIC_EVAL_INTERVAL"
printf 'Command: %q ' "${semantic_cmd[@]}"
printf '\n'
run_conda "${semantic_cmd[@]}"

log_stage "Train Acoustic Decoder"
acoustic_cmd=(
  python -u train.py config/train_audio_acoustic_decoder_v5.py
  "--dataset=$(basename "$DATASET_DIR")"
  "--out_dir=$ACOUSTIC_OUT_DIR"
  "--device=$TRAIN_DEVICE"
  "--dtype=$TRAIN_DTYPE"
  "--compile=$TRAIN_COMPILE"
  "--init_from=$ACOUSTIC_INIT_FROM"
  "--block_size=$ACOUSTIC_BLOCK_SIZE"
)
append_train_overrides acoustic_cmd "$ACOUSTIC_MAX_ITERS" "$ACOUSTIC_LR_DECAY_ITERS" "$ACOUSTIC_WARMUP_ITERS" "$ACOUSTIC_EVAL_INTERVAL"
printf 'Command: %q ' "${acoustic_cmd[@]}"
printf '\n'
run_conda "${acoustic_cmd[@]}"

log_stage "Sample End To End"
run_conda python -u sample_audio_v5.py \
  --semantic_out_dir="$SEMANTIC_OUT_DIR" \
  --acoustic_out_dir="$ACOUSTIC_OUT_DIR" \
  --prompt_audio="$PROMPT_AUDIO" \
  --output_wav="$OUTPUT_WAV" \
  --prompt_max_seconds="$PROMPT_MAX_SECONDS" \
  --max_new_seconds="$MAX_NEW_SECONDS" \
  --semantic_source="$SEMANTIC_SOURCE" \
  --device="$TRAIN_DEVICE" \
  --dtype="$TRAIN_DTYPE" \
  --compile="$TRAIN_COMPILE"

log_stage "Done"
echo "Prompt audio: ${OUTPUT_ROOT}_prompt.${OUTPUT_EXT}"
echo "Full audio: $OUTPUT_WAV"
echo "Continuation audio: ${OUTPUT_ROOT}_continuation.${OUTPUT_EXT}"
