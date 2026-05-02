"""
plate_detector.py
=================
Standalone number-plate detection + OCR module.

Uses the YOLOv8 model trained specifically for number plates
(Backend/app/ml/weights/final_best.pth).

Falls back gracefully if weights are missing or ultralytics is absent.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("app.ml.plate_detector")

# ── optional imports ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO          # type: ignore
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False
    _YOLO = None

try:
    import pytesseract as _tess                     # type: ignore
    _TESS_AVAILABLE = True
except Exception:
    _TESS_AVAILABLE = False
    _tess = None

try:
    import easyocr as _easyocr                     # type: ignore
    _EASY_AVAILABLE = True
except Exception:
    _EASY_AVAILABLE = False
    _easyocr = None


# ── public dataclass ─────────────────────────────────────────────────────────
from dataclasses import dataclass, field


@dataclass
class PlateDetection:
    """Result of plate detection on a single image crop."""
    plate_text: str          # e.g. "ABC-123" or ""
    confidence: float        # YOLO box confidence, 0-1
    bbox: tuple[int, int, int, int] = field(default_factory=lambda: (0, 0, 0, 0))
    # (x1, y1, x2, y2) relative to the input crop


# ── main class ───────────────────────────────────────────────────────────────
class PlateDetector:
    """
    Detects number plates in vehicle crops and reads their text via OCR.

    Priority:
      1. YOLOv8 (final_best.pth) + EasyOCR
      2. YOLOv8 + Tesseract
      3. Contour-based heuristic + OCR   (no trained detector)
      4. No-op  (returns empty string)
    """

    def __init__(self, weights_path: str | Path, conf: float = 0.10, device: str = "cpu"):
        self._conf = conf
        self._device = device
        self._model = None
        self._ocr_reader = None
        self._ocr_backend = "none"

        # ── load YOLO plate detector ──────────────────────────────────────
        wp = Path(weights_path)
        if _YOLO_AVAILABLE and wp.exists():
            try:
                self._model = _YOLO(str(wp), task="detect")
                logger.info(f"PlateDetector: loaded YOLO weights from {wp}")
            except Exception as exc:
                logger.warning(f"PlateDetector: could not load {wp}: {exc}")
        else:
            if not wp.exists():
                logger.warning(f"PlateDetector: weights not found at {wp}. Using heuristic fallback.")
            if not _YOLO_AVAILABLE:
                logger.warning("PlateDetector: ultralytics not available. Using heuristic fallback.")

        # ── load OCR ─────────────────────────────────────────────────────
        if _EASY_AVAILABLE:
            try:
                self._ocr_reader = _easyocr.Reader(["en"], gpu=(device == "cuda"), verbose=False)
                self._ocr_backend = "easyocr"
                logger.info("PlateDetector: OCR backend = EasyOCR")
            except Exception as exc:
                logger.warning(f"PlateDetector: EasyOCR init failed: {exc}")

        if self._ocr_backend == "none" and _TESS_AVAILABLE:
            self._ocr_backend = "tesseract"
            logger.info("PlateDetector: OCR backend = Tesseract")

        if self._ocr_backend == "none":
            logger.warning(
                "PlateDetector: no OCR backend available. "
                "Install easyocr (`pip install easyocr`) or pytesseract for text extraction."
            )

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self, vehicle_crop: np.ndarray) -> PlateDetection:
        """
        Run plate detection + OCR on a single vehicle crop (BGR numpy array).
        Returns a PlateDetection with text="" if nothing found.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return PlateDetection(plate_text="", confidence=0.0)

        plate_crop, bbox, conf = self._locate_plate(vehicle_crop)
        if plate_crop is None or plate_crop.size == 0:
            return PlateDetection(plate_text="", confidence=0.0)

        text = self._read_text(plate_crop)
        text = self._clean_plate_text(text)

        return PlateDetection(plate_text=text, confidence=round(conf, 3), bbox=bbox)

    def detect_batch(self, crops: list[np.ndarray]) -> list[PlateDetection]:
        """Process a list of vehicle crops. Returns one PlateDetection per crop."""
        return [self.detect(c) for c in crops]

    # ── internals ────────────────────────────────────────────────────────────

    def _locate_plate(
        self, img: np.ndarray
    ) -> tuple[Optional[np.ndarray], tuple[int, int, int, int], float]:
        """
        Returns (plate_crop, bbox, confidence).
        Tries YOLO first, then contour heuristic.
        """
        if self._model is not None:
            return self._locate_with_yolo(img)
        return self._locate_with_contours(img)

    def _locate_with_yolo(
        self, img: np.ndarray
    ) -> tuple[Optional[np.ndarray], tuple[int, int, int, int], float]:
        try:
            # Try multiple imgsz - plates appear at varied scales in vehicle crops
            best_conf = 0.0
            best_box = None
            for imgsz in [640, 1280, 1920]:
                results = self._model(img, verbose=False, conf=self._conf, imgsz=imgsz)[0]
                for box in results.boxes:
                    c = float(box.conf.item())
                    if c > best_conf:
                        best_conf = c
                        best_box = [int(v) for v in box.xyxy[0]]
                # Early exit on a confident detection
                if best_conf >= 0.5:
                    break

            if best_box:
                x1, y1, x2, y2 = best_box
                # Add padding so OCR sees full plate
                bw, bh = x2 - x1, y2 - y1
                pad_x = max(5, int(bw * 0.2))
                pad_y = max(5, int(bh * 0.3))
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(img.shape[1], x2 + pad_x)
                y2 = min(img.shape[0], y2 + pad_y)
                crop = img[y1:y2, x1:x2]
                return crop, (x1, y1, x2, y2), best_conf
        except Exception as exc:
            logger.debug(f"YOLO plate detection error: {exc}")
        return None, (0, 0, 0, 0), 0.0

    def _locate_with_contours(
        self, img: np.ndarray
    ) -> tuple[Optional[np.ndarray], tuple[int, int, int, int], float]:
        """
        Simple heuristic: find rectangular regions in lower half of image
        that match typical plate aspect ratios (2:1 to 6:1).
        """
        try:
            h, w = img.shape[:2]
            roi = img[h // 2:, :]   # lower half more likely to contain plate

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_crop = None
            best_box = (0, 0, 0, 0)
            best_score = 0.0

            for cnt in contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                if rw < 40 or rh < 12:
                    continue
                aspect = rw / max(rh, 1)
                if not (1.8 <= aspect <= 7.0):
                    continue
                area_score = (rw * rh) / (w * h / 2)
                if area_score > best_score:
                    best_score = area_score
                    bx1, by1 = max(0, rx), max(0, h // 2 + ry)
                    bx2, by2 = min(w, rx + rw), min(h, h // 2 + ry + rh)
                    best_crop = img[by1:by2, bx1:bx2]
                    best_box = (bx1, by1, bx2, by2)

            if best_crop is not None and best_crop.size > 0:
                return best_crop, best_box, min(0.4, best_score * 2)
        except Exception as exc:
            logger.debug(f"Contour plate detection error: {exc}")
        return None, (0, 0, 0, 0), 0.0

    def _read_text(self, plate_crop: np.ndarray) -> str:
        """OCR the plate crop."""
        if plate_crop is None or plate_crop.size == 0:
            return ""

        # Pre-process for better OCR
        processed = self._preprocess_for_ocr(plate_crop)

        if self._ocr_backend == "easyocr":
            return self._ocr_easyocr(processed)
        elif self._ocr_backend == "tesseract":
            return self._ocr_tesseract(processed)
        return ""

    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Upscale + sharpen + binarize for better OCR accuracy."""
        try:
            h, w = img.shape[:2]
            # Upscale small plates
            if w < 200:
                scale = 200 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Adaptive threshold handles varied lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            # Slight dilation to connect characters
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
            return binary
        except Exception:
            return img

    def _ocr_easyocr(self, img: np.ndarray) -> str:
        try:
            results = self._ocr_reader.readtext(img, detail=1, paragraph=False)
            if not results:
                return ""
            # Pick highest-confidence result
            results.sort(key=lambda r: r[2], reverse=True)
            return results[0][1]
        except Exception as exc:
            logger.debug(f"EasyOCR error: {exc}")
            return ""

    def _ocr_tesseract(self, img: np.ndarray) -> str:
        try:
            config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            text = _tess.image_to_string(img, config=config)
            return text.strip()
        except Exception as exc:
            logger.debug(f"Tesseract error: {exc}")
            return ""

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """Normalize plate text: uppercase, remove noise characters."""
        if not text:
            return ""
        text = text.upper().strip()
        # Keep only alphanumerics and hyphens
        text = re.sub(r"[^A-Z0-9\-]", "", text)
        # Remove very short/noisy results
        if len(text) < 3:
            return ""
        return text


# ── module-level singleton ───────────────────────────────────────────────────
_plate_detector_instance: Optional[PlateDetector] = None


def get_plate_detector(weights_path: str | Path, device: str = "cpu") -> PlateDetector:
    global _plate_detector_instance
    if _plate_detector_instance is None:
        _plate_detector_instance = PlateDetector(weights_path=weights_path, device=device)
    return _plate_detector_instance