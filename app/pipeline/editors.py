import random

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def difficulty_factor(difficulty: str) -> float:
    mapping = {
        "easy": 1.3,
        "medium": 1.0,
        "hard": 0.75,
    }
    return mapping[difficulty]


def apply_random_edit(
    region: Image.Image,
    rng: random.Random,
    strength_scale: float = 1.0,
    preferred_mode: str | None = None,
) -> tuple[Image.Image, str, float]:
    mode = preferred_mode or rng.choice(["brightness", "color", "flip"])

    if mode == "brightness":
        delta = rng.uniform(-0.35, 0.35) * strength_scale
        factor = max(0.65, min(1.35, 1.0 + delta))
        return ImageEnhance.Brightness(region).enhance(factor), mode, factor

    if mode == "color":
        delta = rng.uniform(-0.7, 0.7) * strength_scale
        factor = max(0.5, min(1.7, 1.0 + delta))
        return ImageEnhance.Color(region).enhance(factor), mode, factor

    return ImageOps.mirror(region), mode, 1.0


def blend_region_with_feather(
    base_region: Image.Image,
    edited_region: Image.Image,
    feather_radius: int,
    edit_mask: Image.Image | None = None,
) -> Image.Image:
    width, height = base_region.size

    # Edge mask avoids hard seams along the rectangular crop border.
    edge_mask = Image.new("L", (width, height), 0)
    inset = max(1, feather_radius * 2)
    draw = ImageDraw.Draw(edge_mask)
    draw.rectangle((inset, inset, max(inset + 1, width - inset), max(inset + 1, height - inset)), fill=255)
    edge_mask = edge_mask.filter(ImageFilter.GaussianBlur(radius=max(1, feather_radius)))

    if edit_mask is None:
        final_mask = edge_mask
    else:
        final_mask = ImageChops.multiply(edit_mask.convert("L"), edge_mask)

    return Image.composite(edited_region, base_region, final_mask)


def create_natural_edit_mask(
    size: tuple[int, int],
    rng: random.Random,
    min_coverage: float = 0.25,
    max_coverage: float = 0.65,
) -> tuple[Image.Image, float]:
    width, height = size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    target_coverage = rng.uniform(min_coverage, max_coverage)
    shape_count = rng.randint(2, 5)

    for _ in range(shape_count):
        rw = rng.randint(max(8, width // 6), max(9, int(width * 0.7)))
        rh = rng.randint(max(8, height // 6), max(9, int(height * 0.7)))
        x0 = rng.randint(0, max(0, width - rw))
        y0 = rng.randint(0, max(0, height - rh))
        x1 = x0 + rw
        y1 = y0 + rh

        if rng.random() < 0.5:
            draw.ellipse((x0, y0, x1, y1), fill=255)
        else:
            draw.rounded_rectangle((x0, y0, x1, y1), radius=max(2, min(rw, rh) // 6), fill=255)

    # Light blur creates more organic boundaries.
    blur_radius = max(1, min(width, height) // 18)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Estimate actual mask coverage after blur.
    values = list(mask.getdata())
    coverage = sum(v for v in values) / (255.0 * len(values))

    # If too small, fallback to a centered soft ellipse to ensure visible changes.
    if coverage < target_coverage * 0.6:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        mx = max(4, width // 8)
        my = max(4, height // 8)
        draw.ellipse((mx, my, width - mx, height - my), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        values = list(mask.getdata())
        coverage = sum(v for v in values) / (255.0 * len(values))

    return mask, round(coverage, 6)
