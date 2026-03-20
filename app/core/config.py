import os

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

API_KEY = os.getenv("DIFF_APP_API_KEY", "dev-api-key")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "30"))
TRACE_OUTPUT_DIR = os.getenv("TRACE_OUTPUT_DIR", "experiments")
