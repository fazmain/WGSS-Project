"""
Projected Expectations
======================
An interactive digital artwork.

A person stands in front of the camera. The system segments their body,
colours it with a shifting mask, classifies their gender presentation using
body proportions, and surrounds them with floating text labels drawn from
pools of gendered social expectations.

Controls
--------
  q  —  quit
  d  —  toggle debug overlay  (landmarks + segmentation mask thumbnail)
  p  —  toggle presentation mode  (suppresses the classification HUD)

Run
---
  source .venv/bin/activate
  python main.py
"""

import sys
import time
import platform

import cv2

import config


def _find_builtin_camera_index() -> int:
    """
    On macOS, enumerate AVFoundation cameras and return the index of the
    built-in FaceTime HD / built-in camera.  Falls back to config.CAMERA_INDEX
    if detection fails or is not on macOS.
    """
    if platform.system() != "Darwin":
        return config.CAMERA_INDEX

    try:
        import subprocess, json
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True, text=True, timeout=5,
        )
        data    = json.loads(result.stdout)
        cameras = data.get("SPCameraDataType", [])
        names   = [c.get("_name", "") for c in cameras]
        print(f"Detected cameras: {names}")

        builtin_idx = next(
            (i for i, n in enumerate(names)
             if "facetime" in n.lower() or "built-in" in n.lower()),
            None,
        )
        if builtin_idx is not None:
            print(f"Using built-in camera '{names[builtin_idx]}' at index {builtin_idx}")
            return builtin_idx

        print("Could not identify built-in camera by name; falling back to index 0.")
    except Exception as e:
        print(f"Camera auto-detect error: {e}. Falling back to index {config.CAMERA_INDEX}.")

    return config.CAMERA_INDEX
from src.tracker          import BodyTracker
from src.gender_classifier import GenderClassifier
from src.label_system      import LabelSystem
from src.renderer          import Renderer


# ── Colour map for the HUD category readout (BGR) ────────────────────────────
_HUD_COLORS = {
    "feminine":  (147,  20, 255),   # hot pink
    "masculine": (  0, 200, 255),   # orange
    "mixed":     (  0, 255, 200),   # mint green
}


def main():
    # ── Camera setup ──────────────────────────────────────────────────────────
    cam_idx = _find_builtin_camera_index()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        sys.exit(
            f"Cannot open camera {cam_idx}. "
            "Change CAMERA_INDEX in config.py if you have multiple cameras."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # ── Subsystems ────────────────────────────────────────────────────────────
    tracker    = BodyTracker()
    classifier = GenderClassifier()
    labels     = LabelSystem()
    renderer   = Renderer()

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)

    debug        = False
    presentation = False
    prev_time    = time.perf_counter()

    print("Running — press  q  to quit,  d  for debug,  p  for presentation mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed. Exiting.")
            break

        # Mirror so the display acts like a mirror
        frame = cv2.flip(frame, 1)

        # ── Body tracking + segmentation ──────────────────────────────────────
        landmarks, seg_mask = tracker.process(frame)

        # ── Gender classification ─────────────────────────────────────────────
        category, confidence = classifier.classify(landmarks, frame)

        # ── Timing ────────────────────────────────────────────────────────────
        now       = time.perf_counter()
        dt        = min(now - prev_time, 0.10)   # cap dt to avoid large jumps
        prev_time = now

        # ── Label update ──────────────────────────────────────────────────────
        labels.update(category, confidence, landmarks, dt)

        # ── Render ────────────────────────────────────────────────────────────
        out = renderer.render(
            frame, seg_mask, landmarks, labels.get_labels(), dt,
            debug=debug, presentation=presentation,
        )

        # ── HUD ───────────────────────────────────────────────────────────────
        if not presentation:
            _draw_hud(out, category, confidence, debug)

        cv2.imshow(config.WINDOW_TITLE, out)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            debug = not debug
            print(f"Debug: {'ON' if debug else 'OFF'}")
        elif key == ord("p"):
            presentation = not presentation
            print(f"Presentation mode: {'ON' if presentation else 'OFF'}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


def _draw_hud(frame, category: str, confidence: float, debug: bool):
    """
    Small category readout in the top-right corner.
    Shows the current gender classification and confidence percentage.
    """
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color = _HUD_COLORS.get(category, (200, 200, 200))
    h, w  = frame.shape[:2]

    text = f"{category}  {confidence:.0%}"
    (tw, _), _ = cv2.getTextSize(text, font, 0.72, 2)

    x = w - tw - 20
    y = 36

    # Black outline for legibility on any background
    cv2.putText(frame, text, (x, y), font, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, 0.72, color,    2, cv2.LINE_AA)

    if debug:
        cv2.putText(frame, "DEBUG", (20, 36), font, 0.55,
                    (0, 255, 80), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
