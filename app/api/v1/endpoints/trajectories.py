"""
api/v1/endpoints/trajectories.py
=================================
Cross-video plate matching and trajectory generation.

Endpoints:
  GET /api/v1/trajectories                — list all plates detected in 2+ videos
  GET /api/v1/trajectories/{plate_number} — full journey for one plate across cameras
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import VideoJobORM, VideoResultORM
from app.db.session import get_db
from app.models.schemas import Envelope, ErrorResponse

logger = logging.getLogger("app.api.trajectories")
router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[4]


def _build_waypoints_for_plate(
    plate_number: str, db: Session
) -> tuple[list[dict], list[dict]]:
    """
    Find all videos where this plate was detected.
    Returns (waypoints sorted by upload time, video_summaries).

    Each waypoint = {
        "video_id", "title", "lat", "lng", "name",
        "uploaded_at", "first_seen_in_video", "last_seen_in_video",
        "thumbnail_url"
    }
    """
    # Query all completed videos
    jobs_with_results = (
        db.query(VideoJobORM, VideoResultORM)
        .join(VideoResultORM, VideoJobORM.id == VideoResultORM.job_id)
        .filter(VideoJobORM.status == "completed")
        .all()
    )

    waypoints = []
    video_summaries = []

    for job, result in jobs_with_results:
        if not result.reid_groups:
            continue
        # Find any group in this video with the matching plate
        matching = [
            g for g in result.reid_groups
            if g.get("plate_number") and
            g["plate_number"].upper().strip() == plate_number.upper().strip()
        ]
        if not matching:
            continue

        # Use the group with most detections
        best = max(matching, key=lambda g: g.get("detection_count", 0))

        # Get a thumbnail (first frame from this group)
        thumbnail_url = None
        if result.raw_json:
            detections = result.raw_json.get("detections", [])
            for det in detections:
                if det.get("vehicle_id") == best.get("vehicle_id") and det.get("artifact_path"):
                    art = Path(det["artifact_path"])
                    try:
                        rel = art.relative_to(BASE_DIR / "storage")
                        thumbnail_url = f"/static/storage/{rel}"
                    except ValueError:
                        thumbnail_url = f"/static/storage/videos/{job.id}/artifacts/{art.name}"
                    break

        waypoint = {
            "video_id": job.id,
            "title": job.title,
            "lat": job.camera_lat,
            "lng": job.camera_lng,
            "name": job.camera_name or f"Camera @ {job.title}",
            "uploaded_at": job.created_at.isoformat() if job.created_at else None,
            "first_seen_in_video": best.get("first_seen", 0),
            "last_seen_in_video": best.get("last_seen", 0),
            "detection_count": best.get("detection_count", 0),
            "vehicle_id": best.get("vehicle_id"),
            "thumbnail_url": thumbnail_url,
        }
        waypoints.append(waypoint)
        video_summaries.append({
            "video_id": job.id,
            "title": job.title,
            "vehicle_id": best.get("vehicle_id"),
        })

    # Sort waypoints by upload time (chronological journey)
    waypoints.sort(key=lambda w: w.get("uploaded_at") or "")
    return waypoints, video_summaries


@router.get("", response_model=Envelope)
def list_trajectories(db: Session = Depends(get_db)) -> Envelope:
    """
    List all plates detected in 2 or more videos.
    Returns plates with their journey waypoints.
    """
    # Aggregate plates across all videos
    plate_videos: dict[str, set[int]] = defaultdict(set)
    plate_thumbnails: dict[str, str] = {}

    jobs_with_results = (
        db.query(VideoJobORM, VideoResultORM)
        .join(VideoResultORM, VideoJobORM.id == VideoResultORM.job_id)
        .filter(VideoJobORM.status == "completed")
        .all()
    )

    for job, result in jobs_with_results:
        if not result.reid_groups:
            continue
        for group in result.reid_groups:
            plate = group.get("plate_number")
            if not plate:
                continue
            plate = plate.upper().strip()
            plate_videos[plate].add(job.id)

            # Track first thumbnail we see
            if plate not in plate_thumbnails and result.raw_json:
                detections = result.raw_json.get("detections", [])
                for det in detections:
                    if det.get("vehicle_id") == group.get("vehicle_id") and det.get("artifact_path"):
                        art = Path(det["artifact_path"])
                        try:
                            rel = art.relative_to(BASE_DIR / "storage")
                            plate_thumbnails[plate] = f"/static/storage/{rel}"
                        except ValueError:
                            plate_thumbnails[plate] = f"/static/storage/videos/{job.id}/artifacts/{art.name}"
                        break

    # Filter to plates seen in 2+ videos
    matched_plates = []
    for plate, video_ids in plate_videos.items():
        if len(video_ids) >= 2:
            waypoints, _ = _build_waypoints_for_plate(plate, db)
            # Only include if we have GPS data for all waypoints
            valid_waypoints = [w for w in waypoints if w.get("lat") and w.get("lng")]
            if len(valid_waypoints) >= 2:
                matched_plates.append({
                    "plate_number": plate,
                    "camera_count": len(valid_waypoints),
                    "video_count": len(video_ids),
                    "thumbnail_url": plate_thumbnails.get(plate),
                    "first_seen": min(w.get("uploaded_at") or "" for w in valid_waypoints),
                    "last_seen": max(w.get("uploaded_at") or "" for w in valid_waypoints),
                })

    matched_plates.sort(key=lambda p: p["last_seen"], reverse=True)

    return Envelope(data={
        "plates": matched_plates,
        "total_plates_matched": len(matched_plates),
    })


@router.get("/{plate_number}", response_model=Envelope)
def get_trajectory(plate_number: str, db: Session = Depends(get_db)) -> Envelope:
    """
    Get full journey for a single plate across all videos it was seen in.
    Returns waypoints and video summaries.
    """
    waypoints, video_summaries = _build_waypoints_for_plate(plate_number, db)
    valid_waypoints = [w for w in waypoints if w.get("lat") and w.get("lng")]

    if not valid_waypoints:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorResponse(
                code="not_found",
                message=f"Plate {plate_number} not found in any completed videos with camera locations.",
            ).model_dump(),
        )

    return Envelope(data={
        "plate_number": plate_number.upper().strip(),
        "waypoints": valid_waypoints,
        "video_count": len(video_summaries),
        "camera_count": len(valid_waypoints),
        "is_cross_camera": len(valid_waypoints) >= 2,
    })