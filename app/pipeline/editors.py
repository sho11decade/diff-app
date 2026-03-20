import random

from PIL import Image, ImageEnhance, ImageOps


def difficulty_factor(difficulty: str) -> float:
    mapping = {
        "easy": 1.3,
        "medium": 1.0,
        "hard": 0.75,
    }
    return mapping[difficulty]


def apply_random_edit(region: Image.Image, rng: random.Random) -> tuple[Image.Image, str, float]:
    mode = rng.choice(["brightness", "color", "flip"])

    if mode == "brightness":
        factor = rng.uniform(0.65, 1.35)
        return ImageEnhance.Brightness(region).enhance(factor), mode, factor

    if mode == "color":
        factor = rng.uniform(0.5, 1.7)
        return ImageEnhance.Color(region).enhance(factor), mode, factor

    return ImageOps.mirror(region), mode, 1.0
