from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ── Camera / Location ─────────────────────────────────────────────────────────

class CameraLocation(BaseModel):
    """GPS pin for the camera that recorded this video."""
    lat: float = Field(..., description="Latitude")
    lng: float = Field(..., description="Longitude")
    name: str | None = Field(default=None, max_length=200, description="Human-readable label")


# ── Video Job ─────────────────────────────────────────────────────────────────

class VideoJobBase(BaseModel):
    title: str = Field(..., max_length=255)
    description: str | None = Field(default=None, max_length=2000)


class VideoJobCreate(VideoJobBase):
    camera_location: CameraLocation | None = None


class VideoJob(VideoJobBase):
    id: int
    created_at: datetime
    updated_at: datetime
    status: ProcessingStatus
    original_filename: str
    storage_path: str
    error_message: str | None = None
    progress: int = 0
    duration_ms: int | None = None
    artifact_dir: str | None = None
    log_path: str | None = None
    camera_lat: float | None = None
    camera_lng: float | None = None
    camera_name: str | None = None

    class Config:
        from_attributes = True


class VideoJobListItem(BaseModel):
    id: int
    title: str
    status: ProcessingStatus
    progress: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


# ── Video Result ──────────────────────────────────────────────────────────────

class VideoResult(BaseModel):
    id: int
    job_id: int
    summary: str
    raw_json: dict[str, Any] | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class VideoResultArtifact(BaseModel):
    filename: str
    url: str


class VideoResultWithArtifacts(VideoResult):
    artifacts: list[VideoResultArtifact] | None = None
    metrics: dict[str, Any] | None = None


# ── Re-ID ─────────────────────────────────────────────────────────────────────

class PlateInfo(BaseModel):
    plate_number: str | None = None
    plate_confidence: float = 0.0


class ReIDGroupSchema(BaseModel):
    vehicle_id: str
    detection_count: int
    best_score: float
    first_seen: float
    last_seen: float
    plate_number: str | None = None
    reidentified: bool = False


class VehiclePathSchema(BaseModel):
    vehicle_id: str
    seen_camera_a: bool
    seen_camera_b: bool
    camera_a_time: float | None
    camera_b_time: float | None
    reidentified: bool
    plate_number: str | None = None


class CameraLocationSchema(BaseModel):
    id: str
    name: str
    lat: float
    lng: float


class TrajectorySchema(BaseModel):
    camera_a: CameraLocationSchema
    camera_b: CameraLocationSchema
    vehicle_paths: list[VehiclePathSchema]
    total_vehicles: int
    reidentified_across_cameras: int


class ReIDResultSchema(BaseModel):
    job_id: int
    unique_vehicles: int
    reid_groups: list[ReIDGroupSchema]
    trajectory: TrajectorySchema | dict
    summary: str
    plates_detected: int = 0


# ── Detection (for frames endpoint) ──────────────────────────────────────────

class DetectionFrame(BaseModel):
    url: str
    timestamp: float
    confidence: float
    vehicle_id: str | None
    bbox: list[int]
    plate_number: str | None = None
    plate_confidence: float = 0.0


# ── Errors / Envelope ─────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class Envelope(BaseModel):
    data: Any | None = None
    error: ErrorResponse | None = None