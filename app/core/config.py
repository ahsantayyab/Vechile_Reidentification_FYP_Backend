from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
ENV_PATH   = BASE_DIR / ".env"

if not ENV_PATH.exists():
    raise FileNotFoundError(f".env file not found at {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH, override=True)

ML_WEIGHTS_DIR = BASE_DIR / "app" / "ml" / "weights"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Vehicle Re-identification API"
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    STORAGE_DIR:       Path = BASE_DIR / "storage"
    VIDEO_STORAGE_DIR: Path = BASE_DIR / "storage" / "videos"
    LOG_STORAGE_DIR:   Path = BASE_DIR / "storage" / "logs"

    # ── Model weights ──────────────────────────────────────────────────────────
    # Plate-fused model (best accuracy — copy final_best.pth from training server)
    PLATE_WEIGHTS_PATH: Path | None = ML_WEIGHTS_DIR / "final_best.pth"
    # Base TransReID (fallback if final_best.pth not present)
    MODEL_WEIGHTS_PATH: Path | None = ML_WEIGHTS_DIR / "best_model.pth"

    YOLO_WEIGHTS_PATH:       Path | None = ML_WEIGHTS_DIR / "yolov8n.pt"
    GALLERY_FEATURES_PATH:   Path | None = None
    GALLERY_NAMES_PATH:      Path | None = None

    # ── Detector settings ──────────────────────────────────────────────────────
    DETECTOR_BACKEND:          str        = "yolo"
    DETECTION_CONFIDENCE:      float      = 0.4
    DETECTION_IOU_THRESHOLD:   float      = 0.5
    DETECTION_CLASS_IDS:       list[int] | None = [2, 3, 5, 7]  # car,moto,bus,truck
    DETECTION_MAX_DETECTIONS:  int        = 50
    DETECTION_MIN_BOX_SIZE:    int        = 30

    # ── Processing settings ────────────────────────────────────────────────────
    LOG_LEVEL:             str   = "INFO"
    FRAME_SAMPLING_STRIDE: int   = 5
    MAX_FRAMES_PER_JOB:    int   = 500
    ML_DEVICE:             str   = "auto"
    ML_BATCH_SIZE:         int   = 8

    # ── TransReID architecture ─────────────────────────────────────────────────
    # IMPORTANT: these must match training config exactly
    TRANSREID_NUM_CLASSES:    int        = 777   # VeRi-776 has 777 vehicle IDs
    TRANSREID_NUM_ATTRIBUTES: int        = 2
    TRANSREID_ATTR_CLASSES:   list[int]  = Field(default_factory=lambda: [11, 10])
    TRANSREID_FEATURE_DIM:    int        = 512
    TRANSREID_CAMERA_NUM:     int        = 21    # checkpoint sie_embed shape is [21,1,512]

    # ── Re-ID settings ─────────────────────────────────────────────────────────
    REID_SIMILARITY_THRESHOLD: float = 0.75

    # ── Database ───────────────────────────────────────────────────────────────
    DB_URL: str = Field(..., description="PostgreSQL database URL")

    MAX_UPLOAD_SIZE_MB: int = 500

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    print("✅ DB_URL loaded from .env:", s.DB_URL)
    return s


settings = get_settings()
