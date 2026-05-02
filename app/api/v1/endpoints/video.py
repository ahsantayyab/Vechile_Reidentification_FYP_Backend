"""
api/v1/endpoints/video.py
=========================
Fixed to handle old DB records (no artifact_dir, no plates_detected, etc.)
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.models import VideoJobORM, VideoResultORM
from app.db.session import get_db
from app.models.schemas import Envelope, ErrorResponse, VideoJob, VideoJobListItem

logger = logging.getLogger("app.api.video")
router = APIRouter()

# video.py lives at Backend/app/api/v1/endpoints/video.py
# parents[4] = Backend/  but storage is at VehicleReIdenti/storage (one level up)
BASE_DIR = Path(__file__).resolve().parents[4]


# ── helpers ───────────────────────────────────────────────────────────────────

def _job_or_404(job_id: int, db: Session) -> VideoJobORM:
    job = db.query(VideoJobORM).filter_by(id=job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(code="not_found", message="Job not found.").model_dump(),
        )
    return job


def _artifact_url(rel_path: str) -> str:
    return f"/static/storage/{rel_path}"


def _find_artifacts(job: VideoJobORM) -> list[dict]:
    """Find artifact images on disk, robust to missing/old artifact_dir."""
    items = []

    # Try explicit artifact_dir first
    if job.artifact_dir:
        art_dir = Path(job.artifact_dir)
        if art_dir.exists():
            for img in sorted(art_dir.glob("*.jpg")):
                try:
                    rel = img.relative_to(BASE_DIR / "storage")
                    items.append({"filename": img.name, "url": _artifact_url(str(rel))})
                except ValueError:
                    items.append({
                        "filename": img.name,
                        "url": f"/static/storage/videos/{job.id}/artifacts/{img.name}"
                    })
            return items

    # Fallback: look in standard location
    standard = BASE_DIR / "storage" / "videos" / str(job.id) / "artifacts"
    if standard.exists():
        for img in sorted(standard.glob("*.jpg")):
            items.append({
                "filename": img.name,
                "url": f"/static/storage/videos/{job.id}/artifacts/{img.name}"
            })

    return items


# ── list ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=Envelope)
def list_videos(
    page: int = 1,
    page_size: int = 20,
    status_filter: str | None = None,
    db: Session = Depends(get_db),
) -> Envelope:
    q = db.query(VideoJobORM).order_by(VideoJobORM.created_at.desc())
    if status_filter:
        q = q.filter_by(status=status_filter)
    jobs = q.offset((page - 1) * page_size).limit(page_size).all()
    return Envelope(data=[VideoJobListItem.model_validate(j) for j in jobs])


# ── upload ────────────────────────────────────────────────────────────────────

@router.post("", response_model=Envelope, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str | None = Form(default=None),
    camera_location: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> Envelope:
    # Parse camera location
    cam_lat = cam_lng = cam_name = None
    if camera_location:
        try:
            loc = json.loads(camera_location)
            cam_lat = float(loc.get("lat") or 0) or None
            cam_lng = float(loc.get("lng") or 0) or None
            cam_name = str(loc.get("name") or "").strip() or None
        except Exception as exc:
            logger.warning(f"Could not parse camera_location: {exc}")

    # Save file
    storage_dir = BASE_DIR / "storage" / "videos"
    storage_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    dest = storage_dir / f"{uuid.uuid4().hex}_{safe_name}"
    dest.write_bytes(await file.read())

    # Create job
    job = VideoJobORM(
        title=title,
        description=description,
        status="queued",
        progress=0,
        original_filename=file.filename or "upload.mp4",
        storage_path=str(dest),
        camera_lat=cam_lat,
        camera_lng=cam_lng,
        camera_name=cam_name,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    _enqueue_processing(job.id)

    logger.info(f"Uploaded job_id={job.id} camera=({cam_lat},{cam_lng})")
    return Envelope(data=VideoJob.model_validate(job))


def _enqueue_processing(job_id: int) -> None:
    try:
        from app.services.video_processor import enqueue_job
        enqueue_job(job_id)
    except Exception as exc:
        logger.warning(f"Could not enqueue job {job_id}: {exc}")


# ── get single job ────────────────────────────────────────────────────────────

@router.get("/{job_id}", response_model=Envelope)
def get_video(job_id: int, db: Session = Depends(get_db)) -> Envelope:
    return Envelope(data=VideoJob.model_validate(_job_or_404(job_id, db)))


# ── result ────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/result", response_model=Envelope)
def get_video_result(job_id: int, db: Session = Depends(get_db)) -> Envelope:
    job = _job_or_404(job_id, db)
    result = db.query(VideoResultORM).filter_by(job_id=job_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(code="not_found", message="No result yet.").model_dump(),
        )

    artifacts = _find_artifacts(job)
    raw = result.raw_json or {}
    metrics = dict(raw.get("metrics") or {})

    # Safely get plates_detected (new column, may be None on old rows)
    plates_detected = getattr(result, "plates_detected", None) or 0
    if plates_detected:
        metrics["plates_detected"] = plates_detected

    return Envelope(data={
        "id": result.id,
        "job_id": job_id,
        "summary": result.summary or "",
        "raw_json": raw,
        "created_at": result.created_at.isoformat(),
        "artifacts": artifacts or None,
        "metrics": metrics or None,
    })


# ── frames ────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/frames", response_model=Envelope)
def get_frames(job_id: int, db: Session = Depends(get_db)) -> Envelope:
    job = _job_or_404(job_id, db)
    result = db.query(VideoResultORM).filter_by(job_id=job_id).first()

    if not result or not result.raw_json:
        # Fallback: build frames from artifacts on disk
        artifacts = _find_artifacts(job)
        frames = [{"url": a["url"], "timestamp": 0, "confidence": 0,
                   "vehicle_id": None, "bbox": [], "plate_number": None,
                   "plate_confidence": 0.0} for a in artifacts]
        return Envelope(data={"frames": frames})

    detections = result.raw_json.get("detections") or []
    frames = []
    for det in detections:
        art = det.get("artifact_path")
        if not art:
            continue
        art_path = Path(art)
        try:
            rel = art_path.relative_to(BASE_DIR / "storage")
            url = _artifact_url(str(rel))
        except ValueError:
            url = f"/static/storage/videos/{job_id}/artifacts/{art_path.name}"

        frames.append({
            "url": url,
            "timestamp": det.get("timestamp", 0),
            "confidence": det.get("confidence", 0),
            "vehicle_id": det.get("vehicle_id"),
            "bbox": list(det.get("bbox") or []),
            "plate_number": det.get("plate_number"),
            "plate_confidence": det.get("plate_confidence", 0.0),
        })

    return Envelope(data={"frames": frames})


# ── artifacts ─────────────────────────────────────────────────────────────────

@router.get("/{job_id}/artifacts", response_model=Envelope)
def get_artifacts(job_id: int, db: Session = Depends(get_db)) -> Envelope:
    job = _job_or_404(job_id, db)
    return Envelope(data={"items": _find_artifacts(job)})


# ── logs ──────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/logs", response_model=Envelope)
def get_logs(job_id: int, limit: int = 200, db: Session = Depends(get_db)) -> Envelope:
    job = _job_or_404(job_id, db)
    entries: list[dict] = []
    if job.log_path:
        log_path = Path(job.log_path)
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-limit:]:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    entries.append({"timestamp": 0, "event": "log", "message": line})
    return Envelope(data={"job_id": job_id, "entries": entries})