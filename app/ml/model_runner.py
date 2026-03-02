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

logger = logging.getLogger("app.ml.model_runner")


@dataclass(slots=True)
class DetectionResult:
    timestamp: float
    bbox: tuple[int, int, int, int]
    confidence: float
    matches: list[dict[str, Any]]
    artifact_path: str | None
    vehicle_id: str | None = None   # Re-ID assigned group


@dataclass(slots=True)
class ReIDGroup:
    """A group of detections identified as the same vehicle."""
    vehicle_id: str
    detection_indices: list[int]   # indices into detections list
    best_score: float
    first_seen: float              # timestamp seconds
    last_seen: float
    # For trajectory: camera_a = first appearance, camera_b = last appearance
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
        logger.info("ModelRunner initialized", extra={
            "device": self.config.device,
            "detector": self.detector.name,
        })

    def run(self, video_path: Path, artifacts_dir: Path) -> ModelRunResult:
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

        # ── Re-ID grouping ──────────────────────────────────────────────
        reid_groups: list[dict] = []
        trajectory: dict = {}

        if all_embeddings and len(all_embeddings) > 1:
            reid_groups, trajectory = self._compute_reid_groups(
                detections, all_embeddings
            )
            # Tag each detection with its vehicle_id
            for group in reid_groups:
                for idx in group["detection_indices"]:
                    detections[idx].vehicle_id = group["vehicle_id"]

        elapsed = time.perf_counter() - t0
        summary = self._build_summary(frames_processed, len(detections), len(reid_groups), elapsed)
        metrics = {
            "frames_processed": frames_processed,
            "detections": len(detections),
            "unique_vehicles": len(reid_groups),
            "elapsed_sec": round(elapsed, 2),
            "detector": self.detector.name,
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

    def _flush_batch(self, crops, meta, artifacts_dir):
        embeddings_array = self.feature_extractor.extract(crops)
        embeddings_list = [embeddings_array[i] for i in range(embeddings_array.shape[0])]

        payload: list[DetectionResult] = []
        for det_meta, emb in zip(meta, embeddings_list):
            artifact_path = self._save_snapshot(det_meta, artifacts_dir)
            payload.append(DetectionResult(
                timestamp=round(det_meta.timestamp, 3),
                bbox=det_meta.bbox,
                confidence=round(det_meta.confidence, 3),
                matches=[],
                artifact_path=str(artifact_path) if artifact_path else None,
                vehicle_id=None,
            ))
        return payload, embeddings_list

    def _compute_reid_groups(
        self,
        detections: list[DetectionResult],
        embeddings: list[np.ndarray],
    ) -> tuple[list[dict], dict]:
        """
        Cluster detections by cosine similarity.
        Each cluster = one unique vehicle.
        Returns reid_groups and trajectory data.
        """
        threshold = self.config.reid_similarity_threshold
        n = len(embeddings)
        emb_matrix = np.stack(embeddings)   # (N, 512)

        # Cosine similarity matrix (already L2 normalized)
        sim_matrix = emb_matrix @ emb_matrix.T  # (N, N)

        assigned = [-1] * n
        group_id = 0
        groups: list[list[int]] = []

        for i in range(n):
            if assigned[i] != -1:
                continue
            # Start new group
            groups.append([i])
            assigned[i] = group_id
            # Find all unassigned detections similar to this one
            for j in range(i + 1, n):
                if assigned[j] != -1:
                    continue
                # Compare j against all members of current group
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
            # intra-group similarity score
            if len(indices) > 1:
                pairs = [float(sim_matrix[indices[a], indices[b]])
                         for a in range(len(indices))
                         for b in range(a + 1, len(indices))]
                best_score = float(np.mean(pairs))
            else:
                best_score = 1.0

            reid_groups.append({
                "vehicle_id": vid,
                "detection_indices": indices,
                "detection_count": len(indices),
                "best_score": round(best_score, 3),
                "first_seen": round(min(timestamps), 2),
                "last_seen": round(max(timestamps), 2),
            })

        # ── Trajectory (Camera A → Camera B simulation) ─────────────────
        # We split the video timeline in half:
        # detections in first half = Camera A, second half = Camera B
        trajectory = self._build_trajectory(detections, reid_groups)

        return reid_groups, trajectory

    def _build_trajectory(
        self,
        detections: list[DetectionResult],
        reid_groups: list[dict],
    ) -> dict:
        """
        Simulate a 2-camera trajectory.
        Camera A = Rawalpindi Railway Station (33.5979, 73.0688)
        Camera B = Saddar Rawalpindi        (33.5974, 73.0541)
        """
        if not detections:
            return {}

        all_timestamps = [d.timestamp for d in detections]
        midpoint = (max(all_timestamps) + min(all_timestamps)) / 2

        camera_a = {
            "id": "CAM-A",
            "name": "Camera A – Railway Station",
            "lat": 33.5979,
            "lng": 73.0688,
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

            seen_camera_a = len(cam_a_times) > 0
            seen_camera_b = len(cam_b_times) > 0

            vehicle_paths.append({
                "vehicle_id": group["vehicle_id"],
                "seen_camera_a": seen_camera_a,
                "seen_camera_b": seen_camera_b,
                "camera_a_time": round(min(cam_a_times), 2) if cam_a_times else None,
                "camera_b_time": round(min(cam_b_times), 2) if cam_b_times else None,
                "reidentified": seen_camera_a and seen_camera_b,
            })

        reidentified_count = sum(1 for v in vehicle_paths if v["reidentified"])

        return {
            "camera_a": camera_a,
            "camera_b": camera_b,
            "vehicle_paths": vehicle_paths,
            "total_vehicles": len(reid_groups),
            "reidentified_across_cameras": reidentified_count,
        }

    def _save_snapshot(self, detection: _VehicleDetection, artifacts_dir: Path) -> Path | None:
        snapshot = detection.frame_snapshot.copy()
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(snapshot, f"conf={detection.confidence:.2f}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        filename = f"det_{detection.frame_idx:06d}_{int(detection.timestamp * 1000):07d}.jpg"
        path = artifacts_dir / filename
        cv2.imwrite(str(path), snapshot)
        return path

    def _build_summary(self, frames, detections, vehicles, elapsed) -> str:
        if detections == 0:
            return f"Processed {frames} frames with no vehicle detections in {elapsed:.1f}s."
        return (
            f"Processed {frames} frames, detected {detections} vehicles "
            f"({vehicles} unique identities) in {elapsed:.1f}s."
        )


_runner_instance: ModelRunner | None = None


def get_model_runner() -> ModelRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = ModelRunner()
    return _runner_instance