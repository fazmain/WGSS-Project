"""
src/tracker.py — Body tracking via MediaPipe Pose Landmarker (Tasks API).

Uses the lightweight pose landmarker model which provides:
  • 33-point body landmark detection with visibility scores
  • Optional per-pixel segmentation mask (person vs background)

Model file: models/pose_landmarker_lite.task
  Downloaded once by setup.sh; ~5 MB.

Landmark indices (via mp.tasks.vision.PoseLandmark enum):
  NOSE              = 0
  LEFT_EYE_INNER    = 1   RIGHT_EYE_INNER   = 4
  LEFT_EYE          = 2   RIGHT_EYE         = 5
  LEFT_EYE_OUTER    = 3   RIGHT_EYE_OUTER   = 6
  LEFT_EAR          = 7   RIGHT_EAR         = 8
  MOUTH_LEFT        = 9   MOUTH_RIGHT        = 10
  LEFT_SHOULDER     = 11  RIGHT_SHOULDER    = 12
  LEFT_WRIST        = 15  RIGHT_WRIST       = 16
  LEFT_HIP          = 23  RIGHT_HIP         = 24
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── Model path ─────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

# ── Landmark indices we care about ────────────────────────────────────────────
_PL = mp.tasks.vision.PoseLandmark
_SLOTS = {
    # Face landmarks
    "nose":             _PL.NOSE,
    "left_eye_inner":   _PL.LEFT_EYE_INNER,
    "left_eye":         _PL.LEFT_EYE,
    "left_eye_outer":   _PL.LEFT_EYE_OUTER,
    "right_eye_inner":  _PL.RIGHT_EYE_INNER,
    "right_eye":        _PL.RIGHT_EYE,
    "right_eye_outer":  _PL.RIGHT_EYE_OUTER,
    "left_ear":         _PL.LEFT_EAR,
    "right_ear":        _PL.RIGHT_EAR,
    "mouth_left":       _PL.MOUTH_LEFT,
    "mouth_right":      _PL.MOUTH_RIGHT,
    # Body landmarks (used for label anchors)
    "left_shoulder":    _PL.LEFT_SHOULDER,
    "right_shoulder":   _PL.RIGHT_SHOULDER,
    "left_wrist":       _PL.LEFT_WRIST,
    "right_wrist":      _PL.RIGHT_WRIST,
    "left_hip":         _PL.LEFT_HIP,
    "right_hip":        _PL.RIGHT_HIP,
}


class BodyTracker:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API, VIDEO mode) to expose:
      • landmark pixel positions for named body points
      • a float32 segmentation mask where 1.0 = body, 0.0 = background
    """

    def __init__(self):
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Pose model not found at {_MODEL_PATH}.\n"
                "Run setup.sh (or setup_win.bat) to download it."
            )

        base_opts = mp.tasks.BaseOptions(model_asset_path=str(_MODEL_PATH))
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        self._start_time = time.perf_counter()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray):
        """
        Process one BGR frame.

        Returns
        -------
        landmarks : dict[str, tuple[int,int] | None]
            Pixel coords for each named body point.
            Value is None when the landmark is not visible enough.
        seg_mask  : np.ndarray | None
            Float32 array shape (H, W), values [0, 1].
            1 = body / person,  0 = background.
            None if no person is detected.
        """
        h, w = bgr_frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in ms (must be strictly increasing for VIDEO mode)
        ts_ms = int((time.perf_counter() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        # ── Segmentation mask ──────────────────────────────────────────────────
        seg_mask: Optional[np.ndarray] = None
        if result.segmentation_masks:
            raw = result.segmentation_masks[0].numpy_view()  # float32 H×W
            seg_mask = np.clip(raw, 0.0, 1.0)

        # ── Landmarks ──────────────────────────────────────────────────────────
        landmarks: dict = {}
        if result.pose_landmarks:
            lm_list = result.pose_landmarks[0]  # NormalizedLandmark list

            def to_px(slot):
                lm = lm_list[slot.value]
                if lm.visibility < 0.35:
                    return None
                return (int(lm.x * w), int(lm.y * h))

            landmarks = {name: to_px(slot) for name, slot in _SLOTS.items()}

        return landmarks, seg_mask

    def close(self):
        self._landmarker.close()
