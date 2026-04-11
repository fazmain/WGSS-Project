"""
src/gender_classifier.py — Face-based gender classifier via InsightFace.

Uses the InsightFace buffalo_s model pack:
  • det_500m.onnx   — lightweight face detector
  • genderage.onnx  — gender (M/F) + age regression CNN

The model is downloaded automatically to ~/.insightface/models/buffalo_s/
on first run (~120 MB, one-time download).

Results are smoothed over a rolling window to suppress per-frame flicker.

Returns
-------
category   : 'feminine' | 'masculine' | 'mixed'
confidence : float in [0, 1]  (window fill fraction)
"""

from __future__ import annotations

import os
import warnings
from collections import deque

import numpy as np

import config

# Suppress InsightFace/albumentations SSL warning on first import
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

from insightface.app import FaceAnalysis   # noqa: E402  (import after env setup)


class GenderClassifier:
    """
    Wraps InsightFace's face detection + gender/age model.

    classify() takes the raw BGR frame.  InsightFace internally detects
    the face, aligns it, and runs the gender CNN — no manual crop needed.

    The gender result (0 = female, 1 = male) is mapped to [-1, +1] and
    accumulated in a rolling window, giving a smoothed category.
    """

    def __init__(self):
        self._app = FaceAnalysis(
            name="buffalo_s",
            allowed_modules=["detection", "genderage"],
        )
        # ctx_id=-1 → CPU inference (works everywhere; no GPU required)
        self._app.prepare(ctx_id=-1, det_size=(320, 320))
        self._history: deque[float] = deque(maxlen=config.GENDER_SMOOTH_FRAMES)

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify(self, landmarks: dict, frame: np.ndarray) -> tuple[str, float]:
        """
        Classify the gender presentation in the frame.

        Parameters
        ----------
        landmarks : dict from BodyTracker.process()  (used only for presence check)
        frame     : BGR frame (already mirrored) to detect and classify faces in

        Returns
        -------
        (category_str, confidence_float)
        """
        raw_score = self._detect_gender_score(frame)

        if raw_score is None:
            if not self._history:
                return "mixed", 0.0
        else:
            self._history.append(raw_score)

        if not self._history:
            return "mixed", 0.0

        confidence = len(self._history) / config.GENDER_SMOOTH_FRAMES

        if confidence < config.CONFIDENCE_LOW:
            return "mixed", confidence

        smoothed = float(np.mean(self._history))

        if smoothed > config.MASC_THRESHOLD:
            return "masculine", confidence
        elif smoothed < config.FEM_THRESHOLD:
            return "feminine", confidence
        else:
            return "mixed", confidence

    def reset(self):
        """Clear history — call when the scene goes idle."""
        self._history.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _detect_gender_score(self, frame: np.ndarray):
        """
        Run InsightFace on the frame, pick the largest detected face,
        and return a score in [-1, +1]:
          -1 = female classification
          +1 = male classification
        Returns None if no face is detected.
        """
        faces = self._app.get(frame)
        if not faces:
            return None

        # Use the largest face (most prominent person in frame)
        face = max(faces, key=lambda f: _face_area(f.bbox))

        # face.gender: 0 = female, 1 = male
        return float(face.gender) * 2.0 - 1.0   # map to [-1, +1]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _face_area(bbox) -> float:
    """Return bounding-box area from InsightFace bbox [x1,y1,x2,y2]."""
    return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
