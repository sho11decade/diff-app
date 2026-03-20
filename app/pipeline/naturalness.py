from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageChops, ImageFilter


@dataclass
class NaturalnessMetrics:
    score: float
    mean_abs_diff: float
    edge_delta: float
    change_score: float
    edge_score: float


def evaluate_naturalness(
    original: Image.Image,
    edited: Image.Image,
    target_change: float = 0.12,
    tolerance: float = 0.10,
    edge_penalty_scale: float = 0.25,
) -> NaturalnessMetrics:
    original_rgb = original.convert("RGB")
    edited_rgb = edited.convert("RGB")

    diff = ImageChops.difference(original_rgb, edited_rgb)
    diff_arr = np.asarray(diff, dtype=np.float32) / 255.0
    mean_abs_diff = float(diff_arr.mean())

    edge_original = np.asarray(
        original_rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    ) / 255.0
    edge_edited = np.asarray(
        edited_rgb.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32
    ) / 255.0
    edge_delta = float(np.abs(edge_original - edge_edited).mean())

    # Prefer moderate changes: too small is invisible, too large is unnatural.
    change_score = max(0.0, 1.0 - abs(mean_abs_diff - target_change) / tolerance)

    # Penalize strong edge disruption to keep local structure natural.
    edge_score = max(0.0, 1.0 - edge_delta / edge_penalty_scale)

    score = 0.65 * change_score + 0.35 * edge_score
    return NaturalnessMetrics(
        score=round(score, 6),
        mean_abs_diff=round(mean_abs_diff, 6),
        edge_delta=round(edge_delta, 6),
        change_score=round(change_score, 6),
        edge_score=round(edge_score, 6),
    )
