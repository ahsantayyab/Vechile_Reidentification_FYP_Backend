"""
api/v1/endpoints/reid.py
========================
Returns Re-ID groups + trajectory for a single video.

Now also detects cross-video plate matches:
  - If any plate in this video appears in another video, the trajectory
    response includes `cross_camera_waypoints` listing all cameras the
    matched plate was seen at.
  - The frontend uses this to draw the road path between cameras when
    available, instead of the fake default Camera B.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import VideoJobORM, VideoResultORM
from app.db.session import get_db
from app.models.schemas import Envelope, ErrorResponse

logger = logging.getLogger("app.api.reid")
router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[4]


def _build_thumbnail_url(art_path: str | None, job_id: int) -> str | None:
    if not art_path:
        return None
    p = Path(art_path)
    try:
        rel = p.relative_to(BASE_DIR / "storage")
        return f"/static/storage/{rel}"
    except ValueError:
        return f"/static/storage/videos/{job_id}/artifacts/{p.name}"


def _find_cross_video_matches(
    current_job_id: int, plates_in_current: list[str], db: Session
) -> dict:
    """
    For each plate detected in the current video, find ALL other videos
    where the same plate was detected. Return a dict keyed by plate:

    {
      "ABC-123": [
        {video_id, title, lat, lng, name, uploaded_at, first_seen, ...},
        ...
      ],
      ...
    }

    Only returns plates with cross-video matches (≥2 distinct videos including current).
    """
    if not plates_in_current:
        return {}

    plates_upper = {p.upper().strip() for p in plates_in_current if p}

    # Pull all completed jobs with their results in a single query
    rows = (
        db.query(VideoJobORM, VideoResultORM)
        .join(VideoResultORM, VideoJobORM.id == VideoResultORM.job_id)
        .filter(VideoJobORM.status == "completed")
        .all()
    )

    plate_to_waypoints: dict[str, list[dict]] = defaultdict(list)

    for job, result in rows:
        if not result.reid_groups:
            continue
        if not job.camera_lat or not job.camera_lng:
            continue  # need GPS to plot

        for group in result.reid_groups:
            plate = group.get("plate_number")
            if not plate:
                continue
            plate_norm = plate.upper().strip()
            if plate_norm not in plates_upper:
                continue

            # Get a thumbnail
            thumb = None
            if result.raw_json:
                for det in result.raw_json.get("detections", []):
                    if det.get("vehicle_id") == group.get("vehicle_id") and det.get("artifact_path"):
                        thumb = _build_thumbnail_url(det["artifact_path"], job.id)
                        break

            plate_to_waypoints[plate_norm].append({
                "video_id": job.id,
                "title": job.title,
                "lat": job.camera_lat,
                "lng": job.camera_lng,
                "name": job.camera_name or f"Camera @ {job.title}",
                "uploaded_at": job.created_at.isoformat() if job.created_at else None,
                "first_seen_in_video": group.get("first_seen", 0),
                "last_seen_in_video": group.get("last_seen", 0),
                "detection_count": group.get("detection_count", 0),
                "vehicle_id": group.get("vehicle_id"),
                "thumbnail_url": thumb,
                "is_current_video": job.id == current_job_id,
            })

    # Filter to plates seen in 2+ different videos
    cross_video = {}
    for plate, waypoints in plate_to_waypoints.items():
        unique_video_ids = {w["video_id"] for w in waypoints}
        if len(unique_video_ids) >= 2:
            # Sort by upload time
            waypoints.sort(key=lambda w: w.get("uploaded_at") or "")
            cross_video[plate] = waypoints

    return cross_video


@router.get("/{job_id}/reid", response_model=Envelope)
def get_reid_results(job_id: int, db: Session = Depends(get_db)) -> Envelope:
    """Returns Re-ID groups, trajectory, and cross-video matches for a job."""
    result = db.query(VideoResultORM).filter_by(job_id=job_id).first()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(code="not_found", message="No result for this job.").model_dump(),
        )

    job = db.query(VideoJobORM).filter_by(id=job_id).first()
    reid_groups = result.reid_groups or []
    trajectory = result.trajectory or {}

    # ── Build single-camera location from this video's pin ────────────
    if job and job.camera_lat and job.camera_lng:
        camera_a = {
            "id": "CAM-A",
            "name": job.camera_name or job.title or f"Camera #{job.id}",
            "lat": job.camera_lat,
            "lng": job.camera_lng,
        }
    else:
        camera_a = None

    # ── Detect cross-video plate matches ──────────────────────────────
    plates_in_video = [g.get("plate_number") for g in reid_groups if g.get("plate_number")]
    cross_video_matches = _find_cross_video_matches(job_id, plates_in_video, db)

    # Replace stale trajectory's camera_a with the actual upload pin
    # Also drop the fake Saddar default if this video has no cross-video match
    if camera_a:
        trajectory = {
            "camera_a": camera_a,
            "camera_b": None,                     # only fill if there's a real second camera
            "vehicle_paths": trajectory.get("vehicle_paths", []),
            "total_vehicles": trajectory.get("total_vehicles", len(reid_groups)),
            "reidentified_across_cameras": len(cross_video_matches),
        }
    else:
        trajectory = {}

    return Envelope(data={
        "job_id": job_id,
        "unique_vehicles": result.unique_vehicles,
        "reid_groups": reid_groups,
        "trajectory": trajectory,
        "summary": result.summary,
        "plates_detected": getattr(result, "plates_detected", 0) or 0,
        # Cross-video data: { "ABC-123": [waypoints...] }
        "cross_video_matches": cross_video_matches,
        "has_cross_video_match": bool(cross_video_matches),
    })