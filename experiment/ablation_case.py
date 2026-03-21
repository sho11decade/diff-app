from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter
from statistics import mean

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.generator import generate_differences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ablation case and print JSON metrics.")
    parser.add_argument("--images", nargs="+", required=True, help="Input image paths")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--num-differences", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--label", default="case")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cards = []
    for img_path in args.images:
        image = Image.open(img_path).convert("RGB")
        for s in range(args.seed_offset, args.seed_offset + args.seeds):
            out = generate_differences(
                image=image,
                num_differences=args.num_differences,
                difficulty=args.difficulty,
                seed=s,
            )
            cards.extend(out.difference_cards)

    naturalness = [c.naturalness_score for c in cards]
    gate_pass = [c.score_breakdown.get("quality_gate_passed", 0.0) for c in cards]
    fallback = [1 for c in cards if str(c.edit_type).startswith("fallback_")]
    fg_ratio = [c.score_breakdown.get("region_foreground_ratio", 0.0) for c in cards]

    payload = {
        "label": args.label,
        "samples": len(cards),
        "avg_naturalness": mean(naturalness) if naturalness else 0.0,
        "gate_pass_rate": (sum(gate_pass) / len(gate_pass)) if gate_pass else 0.0,
        "fallback_rate": (sum(fallback) / len(cards)) if cards else 0.0,
        "avg_region_foreground_ratio": mean(fg_ratio) if fg_ratio else 0.0,
        "mode_counts": dict(Counter(c.edit_type for c in cards)),
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
