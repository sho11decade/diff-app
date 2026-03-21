import os

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

API_KEY = os.getenv("DIFF_APP_API_KEY", "dev-api-key")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
TRACE_OUTPUT_DIR = os.getenv("TRACE_OUTPUT_DIR", "experiments")


def _get_bool(name: str, default: bool) -> bool:
	raw = os.getenv(name)
	if raw is None:
		return default
	return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return float(raw)
	except ValueError:
		return default


def _get_int(name: str, default: int) -> int:
	raw = os.getenv(name)
	if raw is None:
		return default
	try:
		return int(raw)
	except ValueError:
		return default


DIFFICULTY_PROFILES: dict[str, dict[str, float]] = {
	"easy": {
		"size_multiplier": _get_float("DIFF_EASY_SIZE_MULTIPLIER", 1.20),
		"initial_strength": _get_float("DIFF_EASY_INITIAL_STRENGTH", 1.30),
		"naturalness_threshold": _get_float("DIFF_EASY_NATURALNESS_THRESHOLD", 0.60),
		"target_change": _get_float("DIFF_EASY_TARGET_CHANGE", 0.18),
		"change_tolerance": _get_float("DIFF_EASY_CHANGE_TOLERANCE", 0.12),
		"min_visible_change": _get_float("DIFF_EASY_MIN_VISIBLE_CHANGE", 0.10),
		"max_visible_change": _get_float("DIFF_EASY_MAX_VISIBLE_CHANGE", 0.28),
		"min_mask_coverage": _get_float("DIFF_EASY_MIN_MASK_COVERAGE", 0.34),
		"min_region_score": _get_float("DIFF_EASY_MIN_REGION_SCORE", 0.28),
	},
	"medium": {
		"size_multiplier": _get_float("DIFF_MEDIUM_SIZE_MULTIPLIER", 1.05),
		"initial_strength": _get_float("DIFF_MEDIUM_INITIAL_STRENGTH", 1.15),
		"naturalness_threshold": _get_float("DIFF_MEDIUM_NATURALNESS_THRESHOLD", 0.58),
		"target_change": _get_float("DIFF_MEDIUM_TARGET_CHANGE", 0.14),
		"change_tolerance": _get_float("DIFF_MEDIUM_CHANGE_TOLERANCE", 0.10),
		"min_visible_change": _get_float("DIFF_MEDIUM_MIN_VISIBLE_CHANGE", 0.08),
		"max_visible_change": _get_float("DIFF_MEDIUM_MAX_VISIBLE_CHANGE", 0.24),
		"min_mask_coverage": _get_float("DIFF_MEDIUM_MIN_MASK_COVERAGE", 0.30),
		"min_region_score": _get_float("DIFF_MEDIUM_MIN_REGION_SCORE", 0.36),
	},
	"hard": {
		"size_multiplier": _get_float("DIFF_HARD_SIZE_MULTIPLIER", 0.95),
		"initial_strength": _get_float("DIFF_HARD_INITIAL_STRENGTH", 1.00),
		"naturalness_threshold": _get_float("DIFF_HARD_NATURALNESS_THRESHOLD", 0.55),
		"target_change": _get_float("DIFF_HARD_TARGET_CHANGE", 0.10),
		"change_tolerance": _get_float("DIFF_HARD_CHANGE_TOLERANCE", 0.08),
		"min_visible_change": _get_float("DIFF_HARD_MIN_VISIBLE_CHANGE", 0.06),
		"max_visible_change": _get_float("DIFF_HARD_MAX_VISIBLE_CHANGE", 0.20),
		"min_mask_coverage": _get_float("DIFF_HARD_MIN_MASK_COVERAGE", 0.26),
		"min_region_score": _get_float("DIFF_HARD_MIN_REGION_SCORE", 0.42),
	},
}


IMPROVEMENT_ATTEMPTS: dict[str, int] = {
	"easy": _get_int("DIFF_EASY_ATTEMPTS", 3),
	"medium": _get_int("DIFF_MEDIUM_ATTEMPTS", 5),
	"hard": _get_int("DIFF_HARD_ATTEMPTS", 7),
}

MIN_DIFF_SIDE = _get_int("DIFF_MIN_SIDE", 32)
DIFF_SIDE_RATIO_MIN = _get_float("DIFF_SIDE_RATIO_MIN", 0.08)
DIFF_SIDE_RATIO_MAX = _get_float("DIFF_SIDE_RATIO_MAX", 0.20)


SEGMENTATION_ENABLED = _get_bool("DIFF_SEGMENTATION_ENABLED", False)
SEGMENTATION_CHECKPOINT_PATH = os.getenv("DIFF_SEGMENTATION_CHECKPOINT", "")
SEGMENTATION_CONFIDENCE_THRESHOLD = _get_float("DIFF_SEGMENTATION_CONFIDENCE_THRESHOLD", 0.50)
SEGMENTATION_REGION_BOOST = _get_float("DIFF_SEGMENTATION_REGION_BOOST", 0.25)
SEGMENTATION_MIN_FOREGROUND_RATIO = _get_float("DIFF_SEGMENTATION_MIN_FOREGROUND_RATIO", 0.06)
SEGMENTATION_IMAGE_SIZE = _get_int("DIFF_SEGMENTATION_IMAGE_SIZE", 512)
SEGMENTATION_DEVICE = os.getenv("DIFF_SEGMENTATION_DEVICE", "")

AB_DENSITY_CONSTRAINT_ENABLED = _get_bool("DIFF_DENSITY_CONSTRAINT_ENABLED", True)
