from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_dir", required=True, help="Directory containing *_metrics.json files.")
    parser.add_argument("--output_path", required=True, help="JSONL manifest path to write.")
    return parser.parse_args()


def score_from_metrics(metrics: dict) -> dict[str, float]:
    critique = metrics.get("critique", {})
    codec = critique.get("codec", {})
    prosody = critique.get("prosody", {})
    semantic = critique.get("semantic", {})

    all_codec_acc = codec.get("all_codebooks", {}).get("overall_accuracy")
    coarse_acc = codec.get("coarse_codebook_0", {}).get("accuracy")
    residual_acc = None
    residual = codec.get("residual_codebooks_1plus")
    if isinstance(residual, dict):
        residual_acc = residual.get("overall_accuracy")

    semantic_acc = semantic.get("reencoded_audio_vs_target", {}).get("accuracy")
    norm_edit = semantic.get("reencoded_audio_vs_target_norm_edit_distance")
    prosody_mae = prosody.get("mean_abs_error")

    realism = float(all_codec_acc if all_codec_acc is not None else 0.0)
    semantic_match = float(semantic_acc if semantic_acc is not None else 0.0)
    intelligibility = float(
        (
            (semantic_acc if semantic_acc is not None else 0.0)
            + (coarse_acc if coarse_acc is not None else 0.0)
            + (1.0 - float(norm_edit) if norm_edit is not None else 0.0)
        ) / 3.0
    )
    prosody_match = float(pow(2.718281828, -(prosody_mae if prosody_mae is not None else 4.0)))
    style_match = float(
        (
            (coarse_acc if coarse_acc is not None else 0.0)
            + (residual_acc if residual_acc is not None else 0.0)
            + prosody_match
        ) / 3.0
    )
    return {
        "realism": max(0.0, min(1.0, realism)),
        "intelligibility": max(0.0, min(1.0, intelligibility)),
        "style_match": max(0.0, min(1.0, style_match)),
        "prosody_match": max(0.0, min(1.0, prosody_match)),
        "semantic_match": max(0.0, min(1.0, semantic_match)),
    }


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    metrics_files = sorted(samples_dir.rglob("*_metrics.json"))
    rows = []
    for metrics_path in metrics_files:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        paths = data.get("paths", {})
        prompt = paths.get("prompt")
        continuation = paths.get("continuation")
        target = paths.get("target_continuation")
        if not prompt or not continuation or not target:
            continue
        row = {
            "prompt_path": prompt,
            "generated_path": continuation,
            "target_path": target,
            "targets": score_from_metrics(data),
            "metrics_path": str(metrics_path),
        }
        rows.append(row)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(rows)} critic examples to {output_path}")


if __name__ == "__main__":
    main()
