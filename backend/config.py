import os
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage"))).resolve()
UPLOADS_DIR = STORAGE_DIR / "uploads"
CLIPS_DIR = STORAGE_DIR / "clips"
TEMP_DIR = STORAGE_DIR / "temp"
JOBS_DIR = STORAGE_DIR / "jobs"
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"

for directory in [STORAGE_DIR, UPLOADS_DIR, CLIPS_DIR, TEMP_DIR, JOBS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Strictly Free-Tier Gemini Models (Google AI Studio Free Tier)
ALLOWED_FREE_GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if DEFAULT_GEMINI_MODEL not in ALLOWED_FREE_GEMINI_MODELS:
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

TARGET_WIDTH = int(os.getenv("TARGET_CLIP_WIDTH", "1080"))
TARGET_HEIGHT = int(os.getenv("TARGET_CLIP_HEIGHT", "1920"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]


class RuntimeConfig:
    """Server-side runtime state for API key management (never exposed to clients)."""

    _gemini_api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    _preferred_model: str = DEFAULT_GEMINI_MODEL

    @classmethod
    def get_api_key(cls) -> str:
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        return cls._gemini_api_key or env_key

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        cls._gemini_api_key = api_key.strip()

    @classmethod
    def has_api_key(cls) -> bool:
        return bool(cls.get_api_key())

    @classmethod
    def get_model(cls) -> str:
        return cls._preferred_model

    @classmethod
    def set_model(cls, model: str) -> None:
        if model not in ALLOWED_FREE_GEMINI_MODELS:
            raise ValueError(
                f"Only free-tier Gemini models are allowed: {', '.join(ALLOWED_FREE_GEMINI_MODELS)}"
            )
        cls._preferred_model = model
