"""
src/renderer.py — Composites the final frame.

The visual language ("the machine gaze"):

  • The subject is poured full of a vertical two-tone gradient — the binary
    pink/blue coding worn openly, as critique.
  • The room becomes a deep ink field: calibration grid, a ghost of the real
    space, film grain, and a scanline that periodically sweeps down and
    "reads" the scene.
  • When the algorithm cannot resolve a binary reading ("mixed"), the image
    itself destabilises: the silhouette drifts through the full spectrum,
    splits into chromatic fringes, and horizontal slices tear loose.
  • Labels are typeset like machine detections — stamped condensed capitals,
    corner brackets, fabricated confidence percentages.
  • A specimen readout in the corner narrates the classification.

Render pipeline (performance-shaped):
  1. Colour fields (background + body) are computed at RENDER_SCALE
     resolution and upscaled once — gradients, glow and grain survive
     the upscale untouched.
  2. Glitch slices tear the full-resolution frame.
  3. Typography is drawn straight onto the frame through PIL's RGBA draw
     mode (only glyph pixels are touched), with per-label pre-rendered
     tiles cached for reuse.
  4. Optional debug overlay.
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32)


def _hsv_to_rgb(h_deg: float, s: float, v: float) -> np.ndarray:
    """h in degrees, s/v in [0,1] → RGB float array 0–255."""
    px = np.array([[[h_deg / 2.0, s * 255.0, v * 255.0]]], dtype=np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    return bgr[::-1].copy()   # BGR → RGB


def _load_font(candidates, size: int) -> ImageFont.FreeTypeFont:
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


# ── Renderer ───────────────────────────────────────────────────────────────────

class Renderer:
    def __init__(self):
        self._elapsed = 0.0
        self._rng = np.random.default_rng()

        # Palette state (RGB float 0–255) — glides toward the target category
        start = config.PALETTES["mixed"]
        self._body_top    = _hex_to_rgb(start["body_top"])
        self._body_bottom = _hex_to_rgb(start["body_bottom"])
        self._background  = _hex_to_rgb(start["background"])
        self._accent      = _hex_to_rgb(start["accent"])

        # 0 → binary reading, 1 → unresolved (drives the glitch aesthetics)
        self._glitch_mix = 0.0

        # Idle timer — seconds since a person was last present
        self._absent_time = 99.0

        # Active glitch slices: list of (y0, h, shift, expires_at)
        self._slices: list[tuple[int, int, int, float]] = []

        # Precomputed static maps at field (half) resolution
        self._static_size: Optional[tuple[int, int]] = None
        self._vignette: Optional[np.ndarray] = None   # (h, w, 1) float32
        self._grid: Optional[np.ndarray] = None       # (h, w, 1) float32
        self._ycol: Optional[np.ndarray] = None       # (h, 1) float32
        self._yrow: Optional[np.ndarray] = None       # (h,)  float32 row index

        # Font cache: (role, size) → ImageFont
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

        # Pre-rendered label tiles: key → (tile RGBA, alpha L)
        self._tiles: dict[tuple, tuple[Image.Image, Image.Image]] = {}
        self._tiles_accent_q: Optional[tuple] = None

        # Opacity LUT cache for tile alpha scaling
        self._alpha_luts: dict[int, bytes] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        seg_mask: Optional[np.ndarray],
        landmarks: dict,
        labels: list,
        dt: float,
        category: str = "mixed",
        confidence: float = 0.0,
        debug: bool = False,
        presentation: bool = False,
    ) -> np.ndarray:
        self._elapsed += dt
        fh, fw = frame.shape[:2]
        s = config.RENDER_SCALE
        w2, h2 = max(2, int(fw * s)), max(2, int(fh * s))
        self._ensure_static(w2, h2)

        person_present = bool(landmarks)
        self._absent_time = 0.0 if person_present else self._absent_time + dt

        self._step_palette(category, dt)

        # ── 1. Colour fields at reduced resolution ────────────────────────────
        frame_small = cv2.resize(frame, (w2, h2), interpolation=cv2.INTER_LINEAR)
        field = self._background_field(frame_small)

        if seg_mask is not None:
            mask_small = cv2.resize(seg_mask, (w2, h2), interpolation=cv2.INTER_LINEAR)
            field = self._body_field(field, mask_small)

        field_u8 = np.clip(field, 0, 255).astype(np.uint8)
        out = cv2.resize(field_u8, (fw, fh), interpolation=cv2.INTER_LINEAR)

        # ── 2. Glitch slices (unresolved readings tear the image) ─────────────
        if self._glitch_mix > 0.5:
            out = self._apply_slices(out)

        # ── 3. Typography ─────────────────────────────────────────────────────
        out = self._text_layer(
            out, labels, category, confidence,
            presentation=presentation, person_present=person_present,
        )

        # ── 4. Debug overlay ──────────────────────────────────────────────────
        if debug:
            _draw_debug(out, landmarks, seg_mask)

        return out

    # ── Palette ────────────────────────────────────────────────────────────────

    def _step_palette(self, category: str, dt: float):
        pal = config.PALETTES.get(category, config.PALETTES["mixed"])

        if category == "mixed":
            # Unresolved: the silhouette drifts through the full spectrum
            hue = (self._elapsed * 22.0) % 360.0
            tgt_top    = _hsv_to_rgb(hue, 0.72, 0.96)
            tgt_bottom = _hsv_to_rgb((hue + 95.0) % 360.0, 0.78, 0.88)
        else:
            tgt_top    = _hex_to_rgb(pal["body_top"])
            tgt_bottom = _hex_to_rgb(pal["body_bottom"])

        tgt_bg     = _hex_to_rgb(pal["background"])
        tgt_accent = _hex_to_rgb(pal["accent"])
        tgt_glitch = 1.0 if category == "mixed" else 0.0

        # Exponential approach — reaches ~95% in PALETTE_LERP_TIME seconds
        k = 1.0 - np.exp(-3.0 * dt / max(config.PALETTE_LERP_TIME, 1e-3))
        self._body_top    += (tgt_top    - self._body_top)    * k
        self._body_bottom += (tgt_bottom - self._body_bottom) * k
        self._background  += (tgt_bg     - self._background)  * k
        self._accent      += (tgt_accent - self._accent)      * k
        self._glitch_mix  += (tgt_glitch - self._glitch_mix)  * k

    # ── Static maps (field resolution) ─────────────────────────────────────────

    def _ensure_static(self, w: int, h: int):
        if self._static_size == (w, h):
            return
        self._static_size = (w, h)
        s = config.RENDER_SCALE

        # Vignette — quadratic radial falloff
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
        vig = 1.0 - config.BG_VIGNETTE * np.clip(r, 0, 1.4) ** 2
        self._vignette = vig[..., np.newaxis].astype(np.float32)

        # Calibration grid + registration ticks at intersections
        grid = np.zeros((h, w), dtype=np.float32)
        gs = max(8, int(config.BG_GRID_SPACING * s))
        grid[::gs, :] = 1.0
        grid[:, ::gs] = 1.0
        tick = max(2, int(5 * s))
        for gy in range(0, h, gs * 2):
            for gx in range(0, w, gs * 2):
                grid[max(0, gy - tick):gy + tick + 1, gx] = 2.4
                grid[gy, max(0, gx - tick):gx + tick + 1] = 2.4
        self._grid = grid[..., np.newaxis]

        # Row indices for the body gradient / shimmer
        self._yrow = np.arange(h, dtype=np.float32)
        self._ycol = (self._yrow / max(h - 1, 1))[:, np.newaxis]

    # ── Background ─────────────────────────────────────────────────────────────

    def _background_field(self, frame_small: np.ndarray) -> np.ndarray:
        h, w = frame_small.shape[:2]
        s = config.RENDER_SCALE
        bg_bgr = self._background[::-1]        # work in BGR to match OpenCV
        acc_bgr = self._accent[::-1]

        # Ink base shaped by the vignette
        out = self._vignette * bg_bgr[np.newaxis, np.newaxis, :]

        # Ghost of the real room — a memory of the space behind the reading
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out += gray[..., np.newaxis] * config.BG_GHOST_ALPHA

        # Calibration grid, tinted with the accent
        out += self._grid * (acc_bgr[np.newaxis, np.newaxis, :] * config.BG_GRID_ALPHA)

        # Scanline sweep — the system periodically re-reads the scene
        t_cycle = self._elapsed % config.SCANLINE_PERIOD
        y_line = t_cycle * config.SCANLINE_SPEED * s
        trail = int(90 * s)
        if y_line < h + trail:
            y0 = int(max(0, y_line - trail))
            y1 = int(min(h, y_line))
            if y1 > y0:
                fall = np.linspace(0.0, 1.0, y1 - y0, dtype=np.float32) ** 2
                band = fall[:, np.newaxis, np.newaxis] * config.SCANLINE_STRENGTH
                out[y0:y1] += band * (acc_bgr * 0.6 + 100.0)

        # Film grain
        noise = self._rng.standard_normal((h // 2, w // 2), dtype=np.float32) \
            * config.BG_GRAIN
        out += cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]

        return out

    # ── Body ───────────────────────────────────────────────────────────────────

    def _body_field(self, out: np.ndarray, seg_mask: np.ndarray) -> np.ndarray:
        h, w = out.shape[:2]
        s = config.RENDER_SCALE

        binary = (seg_mask > 0.5).astype(np.float32)
        soft = cv2.GaussianBlur(binary, (7, 7), 0)
        alpha = soft[..., np.newaxis]

        # Vertical two-tone gradient with a slow shimmer moving up the body
        top_bgr    = self._body_top[::-1]
        bottom_bgr = self._body_bottom[::-1]
        col = top_bgr[np.newaxis, :] + (bottom_bgr - top_bgr)[np.newaxis, :] * self._ycol
        shimmer = 1.0 + config.BODY_SHIMMER_STRENGTH * np.sin(
            2.0 * np.pi * (
                self._yrow / (config.BODY_SHIMMER_PERIOD * s)
                + self._elapsed * config.BODY_SHIMMER_SPEED
            )
        ).astype(np.float32)
        col = col * shimmer[:, np.newaxis]
        body = col[:, np.newaxis, :]           # (h, 1, 3) broadcasts over width

        # Composite — chromatic fringing splits the channels when unresolved
        shift = int(round(config.CHROMA_SHIFT_PX * s * self._glitch_mix))
        if shift > 0:
            a_r = np.roll(soft, -shift, axis=1)[..., np.newaxis]
            a_b = np.roll(soft,  shift, axis=1)[..., np.newaxis]
            out_b = out[..., 0:1] * (1 - a_b) + body[..., 0:1] * a_b
            out_g = out[..., 1:2] * (1 - alpha) + body[..., 1:2] * alpha
            out_r = out[..., 2:3] * (1 - a_r) + body[..., 2:3] * a_r
            out = np.concatenate([out_b, out_g, out_r], axis=2)
        else:
            out = out * (1.0 - alpha) + body * alpha

        # Grain inside the silhouette — a printed, physical texture
        noise = self._rng.standard_normal((h // 2, w // 2), dtype=np.float32) \
            * config.BODY_GRAIN
        out += cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)[..., np.newaxis] * alpha

        # Outer glow — the subject radiates into the ink
        small = cv2.resize(binary, (w // 2, h // 2))
        halo_s = cv2.GaussianBlur(small, (0, 0), sigmaX=16)
        halo = cv2.resize(halo_s, (w, h))
        halo = np.clip(halo - soft, 0.0, 1.0)[..., np.newaxis]
        glow_color = (self._body_top[::-1] + self._body_bottom[::-1]) / 2.0
        out += halo * glow_color[np.newaxis, np.newaxis, :] * config.BODY_GLOW_STRENGTH

        # Thin bright rim on the silhouette edge — softened so the half-res
        # upscale doesn't leave a staircased outline
        b_u8 = (binary * 255).astype(np.uint8)
        rim = cv2.morphologyEx(b_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        rim = cv2.GaussianBlur(rim, (5, 5), 0)
        rim_f = (rim.astype(np.float32) / 255.0)[..., np.newaxis] * config.BODY_RIM_STRENGTH
        rim_color = self._accent[::-1] * 0.55 + 255.0 * 0.45
        out = out * (1.0 - rim_f) + rim_color[np.newaxis, np.newaxis, :] * rim_f

        return out

    # ── Glitch slices ──────────────────────────────────────────────────────────

    def _apply_slices(self, img: np.ndarray) -> np.ndarray:
        h = img.shape[0]

        self._slices = [sl for sl in self._slices if sl[3] > self._elapsed]
        if random.random() < config.GLITCH_SLICE_CHANCE * self._glitch_mix:
            y0 = random.randint(0, h - 90)
            band = random.randint(18, 80)
            shift = random.randint(8, config.GLITCH_MAX_SHIFT) * random.choice((-1, 1))
            self._slices.append((y0, band, shift, self._elapsed + random.uniform(0.06, 0.18)))

        for y0, band, shift, _ in self._slices:
            img[y0:y0 + band] = np.roll(img[y0:y0 + band], shift, axis=1)
        return img

    # ── Typography ─────────────────────────────────────────────────────────────

    def _font(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        key = (role, size)
        if key not in self._fonts:
            cands = (config.LABEL_FONT_CANDIDATES if role == "label"
                     else config.MONO_FONT_CANDIDATES)
            self._fonts[key] = _load_font(cands, size)
        return self._fonts[key]

    def _text_layer(
        self,
        base: np.ndarray,
        labels: list,
        category: str,
        confidence: float,
        presentation: bool,
        person_present: bool,
    ) -> np.ndarray:
        """
        Draw all typography straight onto the frame.

        The BGR numpy frame is wrapped as a PIL image without channel
        conversion — every colour passed to PIL below is therefore given in
        BGR order. RGBA draw mode blends only the touched glyph pixels.
        """
        h, w = base.shape[:2]
        im = Image.fromarray(base)             # BGR bytes, PIL labels it RGB
        d = ImageDraw.Draw(im, "RGBA")

        # Rebuild label tiles when the accent palette has moved enough
        accent_q = tuple(int(c) // 32 for c in self._accent)
        if accent_q != self._tiles_accent_q:
            self._tiles.clear()
            self._tiles_accent_q = accent_q

        for lbl in labels:
            if lbl.pos is None or lbl.opacity < 0.02:
                continue
            self._paste_label(im, lbl, w, h)

        if self._absent_time > 1.5:
            self._draw_idle_prompt(d, w, h)

        if not presentation:
            self._draw_readout(d, w, h, category, confidence, person_present)

        # Title signature, bottom-right
        tfont = self._font("mono", 13)
        title = "PROJECTED EXPECTATIONS"
        tw2 = tfont.getlength(title)
        d.text((w - tw2 - 26, h - 42), title, font=tfont,
               fill=(150, 140, 140, 120))

        arr = np.asarray(im)
        if not arr.flags.writeable:
            arr = arr.copy()
        return arr

    # ── Label tiles ────────────────────────────────────────────────────────────

    def _label_tile(self, lbl) -> tuple[Image.Image, Image.Image, int, int]:
        """Return (tile, alpha, anchor_dx, anchor_dy) for a label, cached."""
        key = (lbl.text, lbl.font_size, lbl.bracketed, lbl.tag)
        cached = self._tiles.get(key)
        if cached:
            return cached

        font = self._font("label", lbl.font_size)
        text = lbl.text.upper()
        tracking = config.LABEL_TRACKING
        tw = self._measure_tracked(text, font, tracking)
        th = lbl.font_size

        pad_x, pad_y = 16, 14
        tag_w = 0
        tfont = None
        if lbl.tag:
            tfont = self._font("mono", 13)
            tag_w = int(tfont.getlength(lbl.tag)) + 10

        tile_w = int(tw) + pad_x * 2 + tag_w
        tile_h = int(th) + pad_y * 2 + 4
        tile = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        td = ImageDraw.Draw(tile)

        acc = self._accent
        # Word colour: near-white pulled toward the accent — stored in BGR
        wc = tuple(int(255 * 0.78 + c * 0.22) for c in acc[::-1])
        acc_bgr = tuple(int(c) for c in acc[::-1])

        x0, y0 = pad_x, pad_y
        # Soft dark shadow, then the word itself
        self._draw_tracked(td, x0 + 2, y0 + 3, text, font, (0, 0, 0, 150), tracking)
        self._draw_tracked(td, x0, y0, text, font, (*wc, 255), tracking)

        # Detection-box corners
        if lbl.bracketed:
            bpad, cl = 9, 9
            bx0, by0 = x0 - bpad, y0 - bpad * 0.6
            bx1, by1 = x0 + tw + bpad, y0 + th + bpad * 0.6
            ca = (*acc_bgr, 210)
            for (cxx, cyy, dx, dy) in (
                (bx0, by0, 1, 1), (bx1, by0, -1, 1),
                (bx0, by1, 1, -1), (bx1, by1, -1, -1),
            ):
                td.line([(cxx, cyy), (cxx + dx * cl, cyy)], fill=ca, width=2)
                td.line([(cxx, cyy), (cxx, cyy + dy * cl)], fill=ca, width=2)

        # Fabricated confidence tag — the machine is always certain
        if lbl.tag:
            td.text((x0 + tw + 8, y0 - 4), lbl.tag, font=tfont,
                    fill=(*acc_bgr, 220))

        record = (tile, tile.getchannel("A"),
                  int(tw / 2) + pad_x, int(th / 2) + pad_y)
        self._tiles[key] = record
        return record

    def _paste_label(self, im: Image.Image, lbl, w: int, h: int):
        pos = lbl.pos + lbl.draw_offset
        x, y = float(pos[0]), float(pos[1])
        if not (-120 < x < w + 120 and -80 < y < h + 80):
            return

        tile, alpha, adx, ady = self._label_tile(lbl)
        box = (int(x) - adx, int(y) - ady)

        op = float(np.clip(lbl.opacity, 0.0, 1.0))
        if op >= 0.99:
            im.paste(tile, box, alpha)
        else:
            lut = self._alpha_lut(op)
            im.paste(tile, box, alpha.point(lut))

    def _alpha_lut(self, opacity: float) -> bytes:
        key = int(opacity * 40)
        lut = self._alpha_luts.get(key)
        if lut is None:
            f = key / 40.0
            lut = bytes(int(v * f) for v in range(256))
            self._alpha_luts[key] = lut
        return lut

    # ── Direct text helpers (BGR colours) ──────────────────────────────────────

    def _draw_tracked(self, d, x, y, text, font, fill, tracking):
        cx = x
        for ch in text:
            d.text((cx, y), ch, font=font, fill=fill)
            cx += font.getlength(ch) + tracking

    def _measure_tracked(self, text, font, tracking) -> float:
        if not text:
            return 0.0
        return sum(font.getlength(ch) for ch in text) + tracking * (len(text) - 1)

    def _draw_idle_prompt(self, d, w, h):
        breathe = 0.55 + 0.40 * np.sin(self._elapsed * 1.5)
        a = float(np.clip(breathe, 0.0, 1.0))
        acc_bgr = tuple(int(c) for c in self._accent[::-1])

        font = self._font("label", 54)
        text = "STEP INTO THE FRAME"
        tracking = 10.0
        tw = self._measure_tracked(text, font, tracking)
        x0 = (w - tw) / 2.0
        y0 = h * 0.42
        self._draw_tracked(d, x0 + 2, y0 + 3, text, font, (0, 0, 0, int(140 * a)), tracking)
        self._draw_tracked(d, x0, y0, text, font, (245, 240, 240, int(235 * a)), tracking)

        sub = "the system is waiting to read you"
        sfont = self._font("mono", 15)
        sw = sfont.getlength(sub)
        d.text(((w - sw) / 2.0, y0 + 78), sub, font=sfont,
               fill=(*acc_bgr, int(190 * a)))

    def _draw_readout(self, d, w, h, category, confidence, person_present):
        mono = self._font("mono", 15)
        acc_bgr = tuple(int(c) for c in self._accent[::-1])
        x, line_h = 26, 24
        y = h - 26 - line_h * 4

        blink = (self._elapsed % 1.2) < 0.72
        dim = (160, 150, 150, 210)
        bright = (240, 235, 235, 235)

        if person_present:
            dot = "●" if blink else "○"
            d.text((x, y), f"{dot} SUBJECT DETECTED", font=mono, fill=(*acc_bgr, 235))
        else:
            dot = "○" if blink else " "
            d.text((x, y), f"{dot} AWAITING SUBJECT", font=mono, fill=dim)
        y += line_h

        if not person_present:
            cls = "—"
        elif category == "mixed":
            cls = "UNRESOLVED" if blink else "UNRESOLVED_"
        else:
            cls = category.upper()
        d.text((x, y), f"CLASS : {cls}", font=mono, fill=bright)
        y += line_h

        # Confidence bar
        d.text((x, y), "CONF  :", font=mono, fill=bright)
        bar_x = x + int(mono.getlength("CONF  : "))
        bar_w, bar_h = 130, 10
        by = y + 5
        d.rectangle([bar_x, by, bar_x + bar_w, by + bar_h],
                    outline=(*acc_bgr, 160), width=1)
        fill_w = int(bar_w * np.clip(confidence, 0, 1))
        if fill_w > 2:
            d.rectangle([bar_x + 2, by + 2, bar_x + fill_w - 2, by + bar_h - 2],
                        fill=(*acc_bgr, 220))
        d.text((bar_x + bar_w + 10, y), f"{confidence:4.0%}", font=mono, fill=bright)
        y += line_h

        d.text((x, y), "MODEL : BINARY (M/F ONLY)", font=mono, fill=dim)


# ── Debug overlay ──────────────────────────────────────────────────────────────

def _draw_debug(
    img: np.ndarray,
    landmarks: dict,
    seg_mask: Optional[np.ndarray],
) -> None:
    """Draw body landmarks and a segmentation mask thumbnail."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img.shape[:2]

    for name, pt in landmarks.items():
        if pt is None:
            continue
        cv2.circle(img, pt, 7, (0, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(img, name[:4], (pt[0] + 8, pt[1] + 4),
                    font, 0.38, (0, 255, 80), 1, cv2.LINE_AA)

    if seg_mask is not None:
        tw, th = w // 6, h // 6
        thumb = cv2.resize((seg_mask * 255).astype(np.uint8), (tw, th))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        img[10:10 + th, 10:10 + tw] = thumb_bgr
        cv2.rectangle(img, (10, 10), (10 + tw, 10 + th), (0, 255, 80), 1)
        cv2.putText(img, "seg mask", (12, 10 + th + 14),
                    font, 0.40, (0, 255, 80), 1, cv2.LINE_AA)
