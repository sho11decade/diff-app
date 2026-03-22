from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, asdict

from PIL import Image, ImageDraw, ImageOps

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.generator import generate_differences


@dataclass
class EvalSummary:
    images: int
    seeds_per_image: int
    num_differences: int
    difficulty: str
    total_cards: int
    avg_naturalness: float
    gate_pass_rate: float
    fallback_rate: float
    mean_effective_visible_change: float
    mean_region_saturation: float
    mean_elapsed_ms: float
    std_elapsed_ms: float
    mode_counts: dict[str, int]


@dataclass
class EvalRaw:
    naturalness_scores: list[float]
    elapsed_ms: list[float]
    effective_visible_changes: list[float]
    region_saturations: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-scale evaluation with free images.")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--seeds-per-image", type=int, default=3)
    parser.add_argument("--num-differences", type=int, default=3)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--source-dir", default="evaluation/free_images_50/source")
    parser.add_argument("--output-dir", default="evaluation/free_images_50/results")
    parser.add_argument("--id-start", type=int, default=1001)
    parser.add_argument("--id-end", type=int, default=1099)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=960)
    parser.add_argument("--preview-count", type=int, default=24, help="Number of preview puzzle/answer pairs to save")
    parser.add_argument("--tile-cols", type=int, default=6, help="Columns in tiled preview image")
    parser.add_argument("--tile-thumb-width", type=int, default=220)
    parser.add_argument("--tile-thumb-height", type=int, default=160)
    return parser.parse_args()


def ensure_images(args: argparse.Namespace) -> tuple[list[pathlib.Path], list[dict[str, str]]]:
    source_dir = pathlib.Path(args.source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(source_dir.glob("*.jpg"))
    manifest_entries: list[dict[str, str]] = []
    for p in existing:
        manifest_entries.append({"file": p.name, "source": "existing"})

    needed = max(0, args.num_images - len(existing))
    if needed == 0:
        return existing[: args.num_images], manifest_entries[: args.num_images]

    downloaded: list[pathlib.Path] = []
    slot = len(existing) + 1
    for img_id in range(args.id_start, args.id_end + 1):
        if len(downloaded) >= needed:
            break
        out_name = f"free_{slot:03d}.jpg"
        out_path = source_dir / out_name
        url = f"https://picsum.photos/id/{img_id}/{args.image_width}/{args.image_height}"
        try:
            urllib.request.urlretrieve(url, out_path)
            Image.open(out_path).verify()
            downloaded.append(out_path)
            manifest_entries.append({"file": out_name, "source": url})
            slot += 1
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)

    all_images = sorted(source_dir.glob("*.jpg"))[: args.num_images]
    if len(all_images) < args.num_images:
        raise RuntimeError(
            f"Not enough images collected: {len(all_images)} / {args.num_images}. "
            "Increase --id-end or reduce --num-images."
        )

    return all_images, manifest_entries[: args.num_images]


def run_eval(
    images: list[pathlib.Path],
    args: argparse.Namespace,
    preview_dir: pathlib.Path,
) -> tuple[EvalSummary, EvalRaw]:
    naturalness_scores: list[float] = []
    gate_flags: list[float] = []
    fallback_flags: list[int] = []
    visible_changes: list[float] = []
    saturations: list[float] = []
    elapsed_ms: list[float] = []
    mode_counter: Counter[str] = Counter()
    saved_previews = 0
    preview_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(images):
        img = Image.open(image_path).convert("RGB")
        seed_base = 10000 + (i * 100)
        for j in range(args.seeds_per_image):
            seed = seed_base + j
            t0 = time.perf_counter()
            out = generate_differences(
                image=img,
                num_differences=args.num_differences,
                difficulty=args.difficulty,
                seed=seed,
            )
            elapsed_ms.append((time.perf_counter() - t0) * 1000.0)

            if saved_previews < args.preview_count and j == 0:
                source_path = preview_dir / f"preview_{saved_previews + 1:03d}_source.jpg"
                puzzle_path = preview_dir / f"preview_{saved_previews + 1:03d}_puzzle.jpg"
                answer_path = preview_dir / f"preview_{saved_previews + 1:03d}_answer.jpg"
                out.source_image.save(source_path, format="JPEG", quality=90)
                out.puzzle_image.save(puzzle_path, format="JPEG", quality=90)
                out.answer_image.save(answer_path, format="JPEG", quality=90)
                saved_previews += 1

            for c in out.difference_cards:
                naturalness_scores.append(c.naturalness_score)
                gate_flags.append(c.score_breakdown.get("quality_gate_passed", 0.0))
                fallback_flags.append(1 if str(c.edit_type).startswith("fallback_") else 0)
                visible_changes.append(c.score_breakdown.get("effective_visible_change", 0.0))
                saturations.append(c.score_breakdown.get("region_mean_saturation", 0.0))
                mode_counter[str(c.edit_type)] += 1

    summary = EvalSummary(
        images=len(images),
        seeds_per_image=args.seeds_per_image,
        num_differences=args.num_differences,
        difficulty=args.difficulty,
        total_cards=len(naturalness_scores),
        avg_naturalness=statistics.mean(naturalness_scores) if naturalness_scores else 0.0,
        gate_pass_rate=(sum(gate_flags) / len(gate_flags)) if gate_flags else 0.0,
        fallback_rate=(sum(fallback_flags) / len(fallback_flags)) if fallback_flags else 0.0,
        mean_effective_visible_change=statistics.mean(visible_changes) if visible_changes else 0.0,
        mean_region_saturation=statistics.mean(saturations) if saturations else 0.0,
        mean_elapsed_ms=statistics.mean(elapsed_ms) if elapsed_ms else 0.0,
        std_elapsed_ms=statistics.pstdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0,
        mode_counts=dict(mode_counter),
    )

    raw = EvalRaw(
        naturalness_scores=naturalness_scores,
        elapsed_ms=elapsed_ms,
        effective_visible_changes=visible_changes,
        region_saturations=saturations,
    )
    return summary, raw


def _draw_histogram(values: list[float], title: str, out_path: pathlib.Path, bins: int = 20) -> None:
    width, height = 960, 480
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 50, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 15), title, fill="black")
    if not values:
        draw.text((20, 40), "no data", fill="gray")
        canvas.save(out_path)
        return

    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    counts = [0] * bins
    for v in values:
        idx = int((v - vmin) / (vmax - vmin) * bins)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_x1 = width - margin_right
    plot_y1 = height - margin_bottom
    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline="#444444", width=1)

    max_count = max(counts) if counts else 1
    bar_w = (plot_x1 - plot_x0) / bins
    for i, c in enumerate(counts):
        x0 = plot_x0 + i * bar_w + 1
        x1 = plot_x0 + (i + 1) * bar_w - 1
        h = 0 if max_count == 0 else (c / max_count) * (plot_y1 - plot_y0)
        y0 = plot_y1 - h
        draw.rectangle((x0, y0, x1, plot_y1), fill="#6c8ebf", outline="#4f6f94")

    draw.text((plot_x0, plot_y1 + 8), f"min={vmin:.4f}", fill="#333333")
    draw.text((plot_x1 - 120, plot_y1 + 8), f"max={vmax:.4f}", fill="#333333")
    draw.text((plot_x0, plot_y0 - 20), f"n={len(values)}", fill="#333333")
    canvas.save(out_path)


def _draw_mode_counts(mode_counts: dict[str, int], title: str, out_path: pathlib.Path) -> None:
    width, height = 960, 540
    margin_left, margin_right, margin_top, margin_bottom = 210, 30, 50, 30
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((20, 15), title, fill="black")
    items = sorted(mode_counts.items(), key=lambda x: x[1], reverse=True)
    if not items:
        draw.text((20, 40), "no data", fill="gray")
        canvas.save(out_path)
        return

    max_v = max(v for _, v in items)
    row_h = min(54, (height - margin_top - margin_bottom) / max(1, len(items)))
    for i, (k, v) in enumerate(items):
        y = margin_top + i * row_h
        draw.text((20, y + 8), k, fill="#222222")
        x0 = margin_left
        x1 = x0 + (0 if max_v == 0 else int((v / max_v) * (width - margin_left - margin_right)))
        draw.rectangle((x0, y + 10, x1, y + row_h - 8), fill="#82b366", outline="#5c8b48")
        draw.text((x1 + 8, y + 8), str(v), fill="#222222")

    canvas.save(out_path)


def _make_preview_tile(preview_dir: pathlib.Path, out_path: pathlib.Path, cols: int, thumb_size: tuple[int, int]) -> None:
    puzzle_paths = sorted(preview_dir.glob("preview_*_puzzle.jpg"))
    if not puzzle_paths:
        return

    rows = math.ceil(len(puzzle_paths) / cols)
    tw, th = thumb_size
    pad = 12
    label_h = 24
    tile = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad), "#f3f3f3")
    draw = ImageDraw.Draw(tile)

    for idx, p in enumerate(puzzle_paths):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tw + pad)
        y = pad + r * (th + label_h + pad)
        img = Image.open(p).convert("RGB")
        fit = ImageOps.fit(img, (tw, th), method=Image.Resampling.LANCZOS)
        tile.paste(fit, (x, y))
        draw.rectangle((x, y, x + tw, y + th), outline="#777777", width=1)
        draw.text((x, y + th + 4), p.stem.replace("_puzzle", ""), fill="#333333")

    tile.save(out_path)


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_outputs"
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    images, manifest_entries = ensure_images(args)
    summary, raw = run_eval(images, args, preview_dir=preview_dir)

    manifest = {
        "provider": "Picsum Photos (random photos from Unsplash contributors)",
        "note": "Check provider terms before redistribution in published materials.",
        "images": manifest_entries,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "evaluation_raw.json").write_text(
        json.dumps(asdict(raw), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _make_preview_tile(
        preview_dir=preview_dir,
        out_path=viz_dir / "preview_tile_puzzle.jpg",
        cols=max(1, args.tile_cols),
        thumb_size=(args.tile_thumb_width, args.tile_thumb_height),
    )
    _draw_mode_counts(summary.mode_counts, "Edit Mode Counts", viz_dir / "mode_counts.png")
    _draw_histogram(raw.naturalness_scores, "Naturalness Score Distribution", viz_dir / "naturalness_hist.png")
    _draw_histogram(raw.elapsed_ms, "Latency Distribution (ms)", viz_dir / "latency_hist.png")

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(f"saved: {output_dir / 'dataset_manifest.json'}")
    print(f"saved: {output_dir / 'evaluation_summary.json'}")
    print(f"saved: {output_dir / 'evaluation_raw.json'}")
    print(f"saved: {viz_dir / 'preview_tile_puzzle.jpg'}")
    print(f"saved: {viz_dir / 'mode_counts.png'}")
    print(f"saved: {viz_dir / 'naturalness_hist.png'}")
    print(f"saved: {viz_dir / 'latency_hist.png'}")


if __name__ == "__main__":
    main()
