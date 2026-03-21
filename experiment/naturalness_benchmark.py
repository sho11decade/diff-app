from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from dataclasses import dataclass
from statistics import mean

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.generator import generate_differences


@dataclass
class BenchmarkResult:
    name: str
    total_cards: int
    mode_counts: dict[str, int]
    avg_naturalness: float
    gate_pass_rate: float
    fallback_rate: float
    mean_effective_visible_change: float
    mean_region_saturation: float


def run_case(name: str, image_path: str, difficulty: str, seeds: int, diffs: int, seed_offset: int) -> BenchmarkResult:
    image = Image.open(image_path).convert("RGB")
    cards = []
    for seed in range(seed_offset, seed_offset + seeds):
        out = generate_differences(
            image=image,
            num_differences=diffs,
            difficulty=difficulty,
            seed=seed,
        )
        cards.extend(out.difference_cards)

    mode_counts = dict(Counter(card.edit_type for card in cards))
    naturalness = [card.naturalness_score for card in cards]
    gate = [card.score_breakdown.get("quality_gate_passed", 0.0) for card in cards]
    fallback = [1 for card in cards if str(card.edit_type).startswith("fallback_")]
    effective = [card.score_breakdown.get("effective_visible_change", 0.0) for card in cards]
    sat = [card.score_breakdown.get("region_mean_saturation", 0.0) for card in cards]

    return BenchmarkResult(
        name=name,
        total_cards=len(cards),
        mode_counts=mode_counts,
        avg_naturalness=mean(naturalness),
        gate_pass_rate=sum(gate) / max(1, len(gate)),
        fallback_rate=sum(fallback) / max(1, len(cards)),
        mean_effective_visible_change=mean(effective),
        mean_region_saturation=mean(sat),
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"=== {result.name} ===")
    print(f"total_cards: {result.total_cards}")
    print(f"mode_counts: {result.mode_counts}")
    print(f"avg_naturalness: {result.avg_naturalness:.4f}")
    print(f"gate_pass_rate: {result.gate_pass_rate:.4f}")
    print(f"fallback_rate: {result.fallback_rate:.4f}")
    print(f"mean_effective_visible_change: {result.mean_effective_visible_change:.4f}")
    print(f"mean_region_saturation: {result.mean_region_saturation:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naturalness generation quality.")
    parser.add_argument("--picture", default="TestImage_Picture.jpg", help="Photo-like test image path")
    parser.add_argument("--illustration", default="TestImage_Illustration.png", help="Illustration-like test image path")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seeds", type=int, default=80)
    parser.add_argument("--diffs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    picture = run_case(
        name="picture",
        image_path=args.picture,
        difficulty=args.difficulty,
        seeds=args.seeds,
        diffs=args.diffs,
        seed_offset=0,
    )
    illustration = run_case(
        name="illustration",
        image_path=args.illustration,
        difficulty=args.difficulty,
        seeds=args.seeds,
        diffs=args.diffs,
        seed_offset=1000,
    )

    print_result(picture)
    print_result(illustration)


if __name__ == "__main__":
    main()
