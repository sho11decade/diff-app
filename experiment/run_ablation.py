from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run segmentation ablation batches.")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--num-differences", type=int, default=4)
    parser.add_argument("--seg-checkpoint", default="", help="Checkpoint path for segmentation integration")
    parser.add_argument("--output", default="experiment/ablation_results.csv")
    return parser.parse_args()


def run_case(
    label: str,
    images: list[str],
    difficulty: str,
    seeds: int,
    num_differences: int,
    extra_env: dict[str, str],
) -> dict:
    env = os.environ.copy()
    env.update(extra_env)

    cmd = [
        sys.executable,
        "experiment/ablation_case.py",
        "--label",
        label,
        "--difficulty",
        difficulty,
        "--seeds",
        str(seeds),
        "--num-differences",
        str(num_differences),
        "--images",
        *images,
    ]

    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(
            f"ablation case failed: {label}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
        )
    return json.loads(cp.stdout.strip())


def main() -> None:
    args = parse_args()

    cases = [
        (
            "baseline_no_seg",
            {
                "DIFF_SEGMENTATION_ENABLED": "0",
            },
        ),
    ]

    if args.seg_checkpoint:
        cases.append(
            (
                "segmentation_enabled",
                {
                    "DIFF_SEGMENTATION_ENABLED": "1",
                    "DIFF_SEGMENTATION_CHECKPOINT": args.seg_checkpoint,
                },
            )
        )

    results = []
    for label, env in cases:
        res = run_case(
            label=label,
            images=args.images,
            difficulty=args.difficulty,
            seeds=args.seeds,
            num_differences=args.num_differences,
            extra_env=env,
        )
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "samples",
                "avg_naturalness",
                "gate_pass_rate",
                "fallback_rate",
                "avg_region_foreground_ratio",
                "mode_counts",
            ],
        )
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["mode_counts"] = json.dumps(row["mode_counts"], ensure_ascii=False)
            writer.writerow(row)

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
