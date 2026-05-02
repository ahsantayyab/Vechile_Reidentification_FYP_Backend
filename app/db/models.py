"""
db/models.py
============
SQLAlchemy ORM models.

Added fields (all nullable / backward-compatible):
  VideoJobORM  : camera_lat, camera_lng, camera_name
  VideoResultORM: plates_detected, reid_groups / trajectory updated via JSON
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer,
    JSON, String, Text
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class VideoJobORM(Base):
    __tablename__ = "video_jobs"

    id               = Column(Integer, primary_key=True, index=True)
    title            = Column(String(255), nullable=False)
    description      = Column(Text, nullable=True)
    status           = Column(String(32), nullable=False, default="queued")
    progress         = Column(Integer, default=0)
    original_filename = Column(String(512), nullable=False)
    storage_path     = Column(String(1024), nullable=False)
    error_message    = Column(Text, nullable=True)
    duration_ms      = Column(Integer, nullable=True)
    artifact_dir     = Column(String(1024), nullable=True)
    log_path         = Column(String(1024), nullable=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                              onupdate=lambda: datetime.now(timezone.utc))

    # ── Camera GPS pin (set at upload time) ──────────────────────────────
    camera_lat  = Column(Float, nullable=True)
    camera_lng  = Column(Float, nullable=True)
    camera_name = Column(String(200), nullable=True)

    result = relationship("VideoResultORM", back_populates="job", uselist=False)


class VideoResultORM(Base):
    __tablename__ = "video_results"

    id              = Column(Integer, primary_key=True, index=True)
    job_id          = Column(Integer, ForeignKey("video_jobs.id"), nullable=False, unique=True)
    summary         = Column(Text, nullable=False)
    raw_json        = Column(JSON, nullable=True)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # ── denormalized Re-ID fields (stored as JSON) ───────────────────────
    unique_vehicles = Column(Integer, default=0)
    reid_groups     = Column(JSON, nullable=True)    # list[dict]
    trajectory      = Column(JSON, nullable=True)    # dict
    plates_detected = Column(Integer, default=0)     # count of detections with a plate

    job = relationship("VideoJobORM", back_populates="result")