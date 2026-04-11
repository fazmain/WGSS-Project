"""
src/renderer.py — Composites the final frame.

Render order:
  1. Mirrored camera frame (base)
  2. Colour-cycling body mask overlay
  3. Soft edge glow on mask boundary
  4. Floating text labels with glow and semi-transparent background
  5. Optional debug overlay (landmarks + segmentation thumbnail)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

import config


class Renderer:
    def __init__(self):
        # Hue cycles continuously for the animated mask colour
        self._hue     = float(config.MASK_BASE_HUE)
        self._elapsed = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        seg_mask: Optional[np.ndarray],
        landmarks: dict,
        labels: list,
        dt: float,
        debug: bool = False,
        presentation: bool = False,
    ) -> np.ndarray:
        """
        Produce one output frame.

        Parameters
        ----------
        frame        : BGR input from webcam (already mirrored)
        seg_mask     : float32 H×W person probability, or None
        landmarks    : dict from BodyTracker
        labels       : list of Label objects from LabelSystem
        dt           : seconds since last frame
        debug        : draw landmarks + segmentation thumbnail
        presentation : suppress HUD (caller handles this)
        """
        self._elapsed += dt
        out = frame.copy()

        # ── Body mask ──────────────────────────────────────────────────────────
        if seg_mask is not None:
            out = self._apply_mask(out, seg_mask, dt)

        # ── Labels ────────────────────────────────────────────────────────────
        for lbl in labels:
            if lbl.pos is None or lbl.opacity < 0.02:
                continue
            _draw_label(out, lbl)

        # ── Debug overlay ─────────────────────────────────────────────────────
        if debug:
            _draw_debug(out, landmarks, seg_mask)

        return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _apply_mask(
        self, frame: np.ndarray, seg_mask: np.ndarray, dt: float
    ) -> np.ndarray:
        """
        Blend a hue-cycling solid colour over the body region.
        Blend a contrasting hue over the background region.
        Also adds a soft glowing edge at the silhouette boundary.
        """
        # Advance hue
        self._hue = (self._hue + config.MASK_HUE_SPEED * dt) % 360.0

        # ── Body colour ───────────────────────────────────────────────────────
        hue_cv    = int(self._hue / 2)   # OpenCV hue: 0–179
        hsv_pixel = np.array(
            [[[hue_cv, config.MASK_SATURATION, config.MASK_VALUE]]],
            dtype=np.uint8,
        )
        bgr_body = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]

        # ── Background colour (offset hue for contrast) ───────────────────────
        bg_hue_cv = int(((self._hue + config.BG_MASK_HUE_OFFSET) % 360.0) / 2)
        hsv_bg    = np.array(
            [[[bg_hue_cv, config.BG_MASK_SATURATION, config.BG_MASK_VALUE]]],
            dtype=np.uint8,
        )
        bgr_bg = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)[0, 0]

        # Threshold the mask to a hard binary silhouette, then smooth only
        # the very edge slightly so the outline isn't jagged.
        binary = (seg_mask > 0.5).astype(np.float32)
        soft_edge = cv2.GaussianBlur(binary, (7, 7), 0)

        body_alpha = soft_edge[..., np.newaxis]          # 1 = body, 0 = background
        bg_alpha   = 1.0 - body_alpha                    # inverse

        frame_f = frame.astype(np.float32)
        body_layer = np.full_like(frame, bgr_body, dtype=np.uint8).astype(np.float32)
        bg_layer   = np.full_like(frame, bgr_bg,   dtype=np.uint8).astype(np.float32)

        # Blend background tint first, then body tint on top
        out = (frame_f * (1.0 - bg_alpha   * config.BG_MASK_ALPHA)
               + bg_layer * (bg_alpha       * config.BG_MASK_ALPHA))
        out = (out    * (1.0 - body_alpha)
               + body_layer * body_alpha)
        out = np.clip(out, 0, 255).astype(np.uint8)

        # ── Edge outline ──────────────────────────────────────────────────────
        edge_map = cv2.Canny((binary * 255).astype(np.uint8), 50, 150)
        edge_dilated = cv2.dilate(edge_map, np.ones((2, 2), np.uint8))
        out[edge_dilated > 0] = (0, 0, 0)

        return out


# ── Module-level drawing helpers ───────────────────────────────────────────────


def _draw_label(img: np.ndarray, lbl) -> None:
    """
    Draw one label with:
      • semi-transparent dark background pill (readability)
      • thin black glow stroke (contrast against any background)
      • coloured text blended at label opacity
    """
    h, w = img.shape[:2]
    x, y = int(lbl.pos[0]), int(lbl.pos[1])

    # Skip labels that are off-screen
    if not (0 < x < w and 10 < y < h):
        return

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = lbl.font_scale
    thickness = config.LABEL_THICKNESS
    alpha     = float(lbl.opacity)
    bgr       = tuple(int(c) for c in lbl.color)

    (tw, th), baseline = cv2.getTextSize(lbl.text, font, scale, thickness)

    # ── Background pill ───────────────────────────────────────────────────────
    pad = 5
    x0, y0 = x - pad, y - th - pad
    x1, y1 = x + tw + pad, y + baseline + pad

    if 0 <= x0 and x1 < w and 0 <= y0 and y1 < h:
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.40, img, 1.0 - alpha * 0.40, 0, img)

    # ── Glow stroke (black, slightly thicker) ─────────────────────────────────
    cv2.putText(img, lbl.text, (x, y), font, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)

    # ── Coloured text blended at opacity ──────────────────────────────────────
    overlay = img.copy()
    cv2.putText(overlay, lbl.text, (x, y), font, scale,
                bgr, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _draw_debug(
    img: np.ndarray,
    landmarks: dict,
    seg_mask: Optional[np.ndarray],
) -> None:
    """Draw body landmarks and a segmentation mask thumbnail."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    # Landmark dots + abbreviated names
    for name, pt in landmarks.items():
        if pt is None:
            continue
        cv2.circle(img, pt, 7, (0, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(img, name[:4], (pt[0] + 8, pt[1] + 4),
                    font, 0.38, (0, 255, 80), 1, cv2.LINE_AA)

    # Segmentation mask thumbnail in top-left corner
    if seg_mask is not None:
        tw, th = w // 6, h // 6
        thumb = cv2.resize((seg_mask * 255).astype(np.uint8), (tw, th))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        img[10:10 + th, 10:10 + tw] = thumb_bgr
        cv2.rectangle(img, (10, 10), (10 + tw, 10 + th), (0, 255, 80), 1)
        cv2.putText(img, "seg mask", (12, 10 + th + 14),
                    font, 0.40, (0, 255, 80), 1, cv2.LINE_AA)
