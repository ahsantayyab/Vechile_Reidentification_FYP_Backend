"""
services/video_processor.py
============================
Background processor that runs the ML pipeline on a queued video job.

Key changes:
  • Reads camera_lat / camera_lng from the job record and passes to model_runner.run()
  • Stores plates_detected count in VideoResultORM
  • Stores full detection list (with plate fields) in raw_json
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import VideoJobORM, VideoResultORM
from app.db.session import SessionLocal
from app.ml.model_runner import get_model_runner

logger = logging.getLogger("app.services.video_processor")

BASE_DIR = Path(__file__).resolve().parents[2]


def process_job(job_id: int) -> None:
    """
    Main entry point called by your task queue / background worker.
    """
    db = SessionLocal()
    try:
        job: VideoJobORM = db.query(VideoJobORM).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in DB.")
            return

        # ── mark processing ───────────────────────────────────────────────
        job.status = "processing"
        job.progress = 5
        db.commit()

        video_path = Path(job.storage_path)
        artifacts_dir = BASE_DIR / "storage" / "videos" / str(job_id) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ── build camera_location from job record ─────────────────────────
        camera_location = None
        if job.camera_lat and job.camera_lng:
            camera_location = {
                "lat": job.camera_lat,
                "lng": job.camera_lng,
                "name": job.camera_name or f"Camera – Job {job_id}",
            }

        # ── run ML pipeline ───────────────────────────────────────────────
        t0 = time.perf_counter()
        runner = get_model_runner()

        job.progress = 20
        db.commit()

        result = runner.run(
            video_path=video_path,
            artifacts_dir=artifacts_dir,
            camera_location=camera_location,
        )

        job.progress = 90
        db.commit()

        # ── serialize ─────────────────────────────────────────────────────
        detections_raw = [asdict(d) for d in result.detections]
        # Convert tuple bboxes to lists for JSON
        for d in detections_raw:
            if isinstance(d.get("bbox"), tuple):
                d["bbox"] = list(d["bbox"])

        raw_json = {
            "summary": result.summary,
            "frames_processed": result.frames_processed,
            "gallery_size": result.gallery_size,
            "metrics": result.metrics,
            "reid_groups": result.reid_groups,
            "trajectory": result.trajectory,
            "detections": detections_raw,
        }

        plates_detected = sum(1 for d in result.detections if d.plate_number)

        # ── save result ───────────────────────────────────────────────────
        db_result = VideoResultORM(
            job_id=job_id,
            summary=result.summary,
            raw_json=raw_json,
            unique_vehicles=len(result.reid_groups),
            reid_groups=result.reid_groups,
            trajectory=result.trajectory,
            plates_detected=plates_detected,
        )
        db.add(db_result)

        # ── update job ────────────────────────────────────────────────────
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        job.status = "completed"
        job.progress = 100
        job.duration_ms = elapsed_ms
        job.artifact_dir = str(artifacts_dir)
        db.commit()

        logger.info(
            f"Job {job_id} completed: {result.frames_processed} frames, "
            f"{len(result.detections)} detections, "
            f"{len(result.reid_groups)} vehicles, "
            f"{plates_detected} plates in {elapsed_ms}ms"
        )

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        try:
            job.status = "failed"
            job.error_message = str(exc)
            db.commit()
        except Exception:
            pass
    finally:
        db.close()


def enqueue_job(job_id: int) -> None:
    """
    Called from the upload endpoint to start processing.
    Uses a simple thread for development; swap for Celery/RQ in production.
    """
    import threading
    t = threading.Thread(target=process_job, args=(job_id,), daemon=True)
    t.start()
    logger.info(f"Enqueued job {job_id} for processing.")