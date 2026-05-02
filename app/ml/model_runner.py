"""
model_runner.py  –  Vehicle Re-ID + Number-Plate detection pipeline
====================================================================
Changes vs original:
  • PlateDetector integrated: runs on every vehicle crop after YOLO detection
  • DetectionResult gains `plate_number` field
  • ReID grouping considers plate text for hard-matching (same non-empty plate
    → same vehicle, regardless of cosine threshold)
  • _build_trajectory gains camera_location support (lat/lng passed in)
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ml.config import DEFAULT_MODEL_CONFIG, ModelConfig
from app.ml.detector import build_detector
from app.ml.feature_extractor import TransReIDFeatureExtractor
from app.ml.gallery_index import GalleryIndex, GalleryMatch
from app.ml.plate_detector import PlateDetector, get_plate_detector

logger = logging.getLogger("app.ml.model_runner")


@dataclass(slots=True)
class DetectionResult:
    timestamp: float
    bbox: tuple[int, int, int, int]
    confidence: float
    matches: list[dict[str, Any]]
    artifact_path: str | None
    vehicle_id: str | None = None      # Re-ID assigned group
    plate_number: str | None = None    # OCR plate text (e.g. "ABC-123")
    plate_confidence: float = 0.0      # YOLO plate detection confidence


@dataclass(slots=True)
class ReIDGroup:
    """A group of detections identified as the same vehicle."""
    vehicle_id: str
    detection_indices: list[int]
    best_score: float
    first_seen: float
    last_seen: float
    plate_number: str | None = None    # consensus plate for the group
    camera_a_timestamp: float = 0.0
    camera_b_timestamp: float = 0.0


@dataclass(slots=True)
class ModelRunResult:
    summary: str
    frames_processed: int
    detections: list[DetectionResult]
    gallery_size: int
    metrics: dict[str, Any]
    reid_groups: list[dict[str, Any]] = field(default_factory=list)
    trajectory: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["detections"] = [asdict(det) for det in self.detections]
        return payload


@dataclass(slots=True)
class _VehicleDetection:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    confidence: float
    frame_idx: int
    timestamp: float
    frame_snapshot: np.ndarray


class ModelRunner:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or DEFAULT_MODEL_CONFIG
        self.feature_extractor = TransReIDFeatureExtractor(self.config)
        self.gallery = GalleryIndex(
            features_path=self.config.gallery_features_path,
            names_path=self.config.gallery_names_path,
        )
        self.detector = build_detector(self.config)

        # ── plate detector ─────────────────────────────────────────────────
        plate_weights = Path(__file__).resolve().parent / "weights" / "license_plate_detector.pt"
        self.plate_detector: PlateDetector | None = get_plate_detector(
            weights_path=plate_weights,
            device=self.config.device,
        )

        logger.info("ModelRunner initialized", extra={
            "device": self.config.device,
            "detector": self.detector.name,
            "plate_detector": "loaded" if self.plate_detector else "disabled",
        })

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        video_path: Path,
        artifacts_dir: Path,
        camera_location: dict | None = None,   # {"lat": float, "lng": float, "name": str}
    ) -> ModelRunResult:
        if not Path(video_path).exists():
            raise FileNotFoundError(video_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = 0
        frames_processed = 0
        pending_crops: list[np.ndarray] = []
        pending_meta: list[_VehicleDetection] = []
        detections: list[DetectionResult] = []
        all_embeddings: list[np.ndarray] = []
        t0 = time.perf_counter()

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_idx % max(1, self.config.frame_stride) != 0:
                    frame_idx += 1
                    continue
                if frames_processed >= self.config.max_frames:
                    break

                frame_dets = self._detect(frame, frame_idx, fps)
                if frame_dets:
                    pending_meta.extend(frame_dets)
                    pending_crops.extend([d.crop for d in frame_dets])
                    if len(pending_crops) >= self.config.batch_size:
                        batch_results, batch_embeddings = self._flush_batch(
                            pending_crops, pending_meta, artifacts_dir
                        )
                        detections.extend(batch_results)
                        all_embeddings.extend(batch_embeddings)
                        pending_crops.clear()
                        pending_meta.clear()

                frames_processed += 1
                frame_idx += 1
        finally:
            capture.release()

        if pending_crops:
            batch_results, batch_embeddings = self._flush_batch(
                pending_crops, pending_meta, artifacts_dir
            )
            detections.extend(batch_results)
            all_embeddings.extend(batch_embeddings)

        # ── Re-ID grouping ──────────────────────────────────────────────────
        reid_groups: list[dict] = []
        trajectory: dict = {}

        if all_embeddings and len(all_embeddings) > 1:
            reid_groups, trajectory = self._compute_reid_groups(
                detections, all_embeddings, camera_location=camera_location
            )
            for group in reid_groups:
                for idx in group["detection_indices"]:
                    detections[idx].vehicle_id = group["vehicle_id"]
                    # Do NOT propagate plate to all frames — only frames that actually
                    # detected the plate keep their plate_number. Group consensus stays in
                    # the reid_group object only.

        elapsed = time.perf_counter() - t0
        summary = self._build_summary(frames_processed, len(detections), len(reid_groups), elapsed)
        metrics = {
            "frames_processed": frames_processed,
            "detections": len(detections),
            "unique_vehicles": len(reid_groups),
            "elapsed_sec": round(elapsed, 2),
            "detector": self.detector.name,
            "plates_detected": sum(1 for d in detections if d.plate_number),
        }

        return ModelRunResult(
            summary=summary,
            frames_processed=frames_processed,
            detections=detections,
            gallery_size=len(reid_groups),
            metrics=metrics,
            reid_groups=reid_groups,
            trajectory=trajectory,
        )

    # ── detection ─────────────────────────────────────────────────────────────

    def _detect(self, frame, frame_idx, fps) -> list[_VehicleDetection]:
        timestamp = frame_idx / max(fps, 1e-3)
        detections = []
        try:
            raw = self.detector.detect(frame)
            for r in raw:
                x1, y1, x2, y2 = r.bbox
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                detections.append(_VehicleDetection(
                    crop=crop, bbox=r.bbox, confidence=r.confidence,
                    frame_idx=frame_idx, timestamp=timestamp,
                    frame_snapshot=frame.copy(),
                ))
        except Exception as exc:
            logger.error("Detector failure", extra={"error": str(exc)})
        return detections

    # ── batch flush ───────────────────────────────────────────────────────────

    def _flush_batch(self, crops, meta, artifacts_dir):
        embeddings_array = self.feature_extractor.extract(crops)
        embeddings_list = [embeddings_array[i] for i in range(embeddings_array.shape[0])]

        # ── plate detection for entire batch ─────────────────────────────
        plate_results = []
        if self.plate_detector:
            try:
                plate_results = self.plate_detector.detect_batch(crops)
            except Exception as exc:
                logger.warning(f"Plate detection batch failed: {exc}")
                plate_results = []

        payload: list[DetectionResult] = []
        for i, (det_meta, emb) in enumerate(zip(meta, embeddings_list)):
            artifact_path = self._save_snapshot(det_meta, artifacts_dir, plate_results[i] if plate_results else None)
            plate_text = plate_results[i].plate_text if plate_results else ""
            plate_conf = plate_results[i].confidence if plate_results else 0.0
            payload.append(DetectionResult(
                timestamp=round(det_meta.timestamp, 3),
                bbox=det_meta.bbox,
                confidence=round(det_meta.confidence, 3),
                matches=[],
                artifact_path=str(artifact_path) if artifact_path else None,
                vehicle_id=None,
                plate_number=plate_text or None,
                plate_confidence=round(plate_conf, 3),
            ))
        return payload, embeddings_list

    # ── Re-ID grouping ────────────────────────────────────────────────────────

    def _compute_reid_groups(
        self,
        detections: list[DetectionResult],
        embeddings: list[np.ndarray],
        camera_location: dict | None = None,
    ) -> tuple[list[dict], dict]:
        threshold = self.config.reid_similarity_threshold
        n = len(embeddings)
        emb_matrix = np.stack(embeddings)
        sim_matrix = emb_matrix @ emb_matrix.T

        assigned = [-1] * n
        group_id = 0
        groups: list[list[int]] = []

        for i in range(n):
            if assigned[i] != -1:
                continue
            groups.append([i])
            assigned[i] = group_id
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                # ── HARD MATCH: identical non-empty plate numbers ────────
                plate_i = detections[i].plate_number
                plate_j = detections[j].plate_number
                if plate_i and plate_j and plate_i == plate_j:
                    groups[group_id].append(j)
                    assigned[j] = group_id
                    continue
                # ── SOFT MATCH: cosine similarity ────────────────────────
                group_indices = groups[group_id]
                scores = [float(sim_matrix[j, k]) for k in group_indices]
                if max(scores) >= threshold:
                    groups[group_id].append(j)
                    assigned[j] = group_id
            group_id += 1

        reid_groups = []
        for gid, indices in enumerate(groups):
            vid = f"VH-{gid + 1:03d}"
            timestamps = [detections[i].timestamp for i in indices]

            if len(indices) > 1:
                pairs = [float(sim_matrix[indices[a], indices[b]])
                         for a in range(len(indices))
                         for b in range(a + 1, len(indices))]
                best_score = float(np.mean(pairs))
            else:
                best_score = 1.0

            # ── consensus plate number for this group ────────────────────
            plates = [detections[i].plate_number for i in indices if detections[i].plate_number]
            consensus_plate = None
            if plates:
                from collections import Counter
                most_common = Counter(plates).most_common(1)
                consensus_plate = most_common[0][0]

            reid_groups.append({
                "vehicle_id": vid,
                "detection_indices": indices,
                "detection_count": len(indices),
                "best_score": round(best_score, 3),
                "first_seen": round(min(timestamps), 2),
                "last_seen": round(max(timestamps), 2),
                "plate_number": consensus_plate,
            })

        trajectory = self._build_trajectory(detections, reid_groups, camera_location)
        return reid_groups, trajectory

    # ── trajectory ────────────────────────────────────────────────────────────

    def _build_trajectory(
        self,
        detections: list[DetectionResult],
        reid_groups: list[dict],
        camera_location: dict | None = None,
    ) -> dict:
        """
        Build trajectory data.
        Camera A = the location pin supplied by the user (or default).
        Camera B = default second camera (simulated for demo).
        """
        if not detections:
            return {}

        all_timestamps = [d.timestamp for d in detections]
        midpoint = (max(all_timestamps) + min(all_timestamps)) / 2

        # User-supplied location for camera A
        if camera_location and camera_location.get("lat") and camera_location.get("lng"):
            cam_a_lat = camera_location["lat"]
            cam_a_lng = camera_location["lng"]
            cam_a_name = camera_location.get("name") or "Camera A – Upload Location"
        else:
            cam_a_lat = 33.5979
            cam_a_lng = 73.0688
            cam_a_name = "Camera A – Railway Station"

        camera_a = {
            "id": "CAM-A",
            "name": cam_a_name,
            "lat": cam_a_lat,
            "lng": cam_a_lng,
        }
        camera_b = {
            "id": "CAM-B",
            "name": "Camera B – Saddar Chowk",
            "lat": 33.5974,
            "lng": 73.0541,
        }

        vehicle_paths = []
        for group in reid_groups:
            indices = group["detection_indices"]
            timestamps = [detections[i].timestamp for i in indices]
            cam_a_times = [t for t in timestamps if t <= midpoint]
            cam_b_times = [t for t in timestamps if t > midpoint]
            seen_a = len(cam_a_times) > 0
            seen_b = len(cam_b_times) > 0

            vehicle_paths.append({
                "vehicle_id": group["vehicle_id"],
                "seen_camera_a": seen_a,
                "seen_camera_b": seen_b,
                "camera_a_time": round(min(cam_a_times), 2) if cam_a_times else None,
                "camera_b_time": round(min(cam_b_times), 2) if cam_b_times else None,
                "reidentified": seen_a and seen_b,
                "plate_number": group.get("plate_number"),
            })

        reidentified_count = sum(1 for v in vehicle_paths if v["reidentified"])

        return {
            "camera_a": camera_a,
            "camera_b": camera_b,
            "vehicle_paths": vehicle_paths,
            "total_vehicles": len(reid_groups),
            "reidentified_across_cameras": reidentified_count,
        }

    # ── snapshot saving ───────────────────────────────────────────────────────

    def _save_snapshot(
        self, detection: _VehicleDetection, artifacts_dir: Path, plate=None
    ) -> Path | None:
        snapshot = detection.frame_snapshot.copy()
        x1, y1, x2, y2 = detection.bbox

        # Vehicle bounding box (green)
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(snapshot, f"conf={detection.confidence:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Plate overlay (cyan) if detected
        if plate and plate.plate_text:
            bx1, by1, bx2, by2 = plate.bbox
            # bbox is relative to crop; offset to frame coords
            abs_bx1, abs_by1 = x1 + bx1, y1 + by1
            abs_bx2, abs_by2 = x1 + bx2, y1 + by2
            if abs_bx2 > abs_bx1 and abs_by2 > abs_by1:
                cv2.rectangle(snapshot, (abs_bx1, abs_by1), (abs_bx2, abs_by2), (255, 255, 0), 2)
            # Plate text label
            label = f"PLATE: {plate.plate_text}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            lx, ly = x1, min(snapshot.shape[0] - 5, y2 + 18)
            cv2.rectangle(snapshot, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 4), (0, 0, 0), -1)
            cv2.putText(snapshot, label, (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        filename = f"det_{detection.frame_idx:06d}_{int(detection.timestamp * 1000):07d}.jpg"
        path = artifacts_dir / filename
        cv2.imwrite(str(path), snapshot)
        return path

    # ── summary ───────────────────────────────────────────────────────────────

    def _build_summary(self, frames, detections, vehicles, elapsed) -> str:
        if detections == 0:
            return f"Processed {frames} frames with no vehicle detections in {elapsed:.1f}s."
        return (
            f"Processed {frames} frames, detected {detections} vehicles "
            f"({vehicles} unique identities) in {elapsed:.1f}s."
        )


# ── singleton ─────────────────────────────────────────────────────────────────
_runner_instance: ModelRunner | None = None


def get_model_runner() -> ModelRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = ModelRunner()
    return _runner_instance