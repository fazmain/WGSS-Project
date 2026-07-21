"""
src/label_system.py — Animated floating text labels.

Each Label:
  • orbits one of several named body anchors
  • drifts with sinusoidal motion on each axis (organic feel)
  • scatters outward when the person moves, then reforms
  • fades in when spawned, fades out when removed
  • follows anchor movement with a configurable spring constant

LabelSystem:
  • spawns / culls labels to match the target count
  • switches label pools when gender category changes
  • estimates distance from shoulder width to adjust label count
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

import config


# ── Anchor slots ───────────────────────────────────────────────────────────────
# Each slot name corresponds to a key in the anchor dict built from landmarks.
# Names are repeated to bias toward certain body regions.
_ANCHOR_SLOTS = [
    "head", "head",
    "left_shoulder", "right_shoulder",
    "torso", "torso",
    "hips", "hips",
    "left_wide", "right_wide",
    "upper_left", "upper_right",
]


class Label:
    """
    A single floating text label.

    Position is animated via:
      1. A sinusoidal drift (different frequency per axis → Lissajous-like curve)
      2. A decaying scatter impulse added on body-movement events
      3. A spring that pulls the label toward (anchor + drift offset)
    """

    def __init__(
        self,
        text: str,
        font_size: int,
        anchor_key: str,
        time_offset: float = 0.0,
        tag: Optional[str] = None,
        bracketed: bool = False,
        glitchy: bool = False,
    ):
        self.text       = text
        self.font_size  = font_size
        self.anchor_key = anchor_key
        self.tag        = tag          # fabricated confidence, e.g. "87%"
        self.bracketed  = bracketed    # drawn with detection-box corners
        self.glitchy    = glitchy      # spawned under an unresolved reading
        self.draw_offset = np.zeros(2) # per-frame glitch jitter

        # ── Drift parameters ──────────────────────────────────────────────────
        # Independent sinusoidal on X and Y gives a smooth orbital feel.
        self._drift_freq  = np.random.uniform(0.12, 0.38, size=2)   # Hz
        self._drift_phase = np.random.uniform(0, 2 * np.pi, size=2)
        self._drift_amp   = np.random.uniform(22, 60, size=2)        # px

        # ── Base offset from anchor ────────────────────────────────────────────
        angle = np.random.uniform(0, 2 * np.pi)
        dist  = np.random.uniform(70, 165)
        self._base_offset = np.array([np.cos(angle) * dist,
                                      np.sin(angle) * dist])

        # ── Scatter physics ───────────────────────────────────────────────────
        self._scatter_vel = np.zeros(2)

        # ── Screen position ───────────────────────────────────────────────────
        self.pos: Optional[np.ndarray] = None   # initialised on first update

        # ── Fade state ────────────────────────────────────────────────────────
        self.opacity    = 0.0
        self.fade_state = "in"   # "in" | "hold" | "out"

        # Labels retire after a while so the vocabulary keeps churning
        self._lifespan = np.random.uniform(8.0, 17.0)

        # Approximate rendered half-extents (for edge clamping / repulsion)
        self._half_w = font_size * 0.26 * len(text) + 16
        self._half_h = font_size * 0.62 + 10

        # Internal time accumulator (stagger prevents synchronised motion)
        self._t = float(time_offset)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        anchors: dict[str, np.ndarray],
        dt: float,
        movement_burst: float,
        person_present: bool,
    ):
        self._t += dt

        # ── Fade ──────────────────────────────────────────────────────────────
        if not person_present:
            self.fade_state = "out"

        if self.fade_state == "in":
            self.opacity = min(1.0, self.opacity + config.LABEL_FADE_SPEED * dt)
            if self.opacity >= 1.0:
                self.fade_state = "hold"
        elif self.fade_state == "hold":
            if self._t > self._lifespan:
                self.fade_state = "out"
        elif self.fade_state == "out":
            self.opacity = max(0.0, self.opacity - config.LABEL_FADE_SPEED * dt)

        # ── Glitch jitter ─────────────────────────────────────────────────────
        # Labels born under an unresolved reading can't be held steady —
        # every couple of seconds they briefly tear a few pixels sideways.
        if self.glitchy and (self._t % 1.9) < 0.10:
            self.draw_offset = np.random.uniform(-9, 9, size=2)
        else:
            self.draw_offset = np.zeros(2)

        # ── Scatter impulse ───────────────────────────────────────────────────
        if movement_burst > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = config.SCATTER_VELOCITY * min(movement_burst / 35.0, 2.5)
            self._scatter_vel += np.array([np.cos(angle), np.sin(angle)]) * speed

        # Exponential decay — half-life ≈ 0.25 s
        self._scatter_vel *= np.exp(-4.0 * dt)

        # ── Target position ───────────────────────────────────────────────────
        anchor = anchors.get(self.anchor_key)
        if anchor is None:
            # Anchor temporarily not visible; keep current position
            return

        drift = (self._drift_amp *
                 np.sin(2 * np.pi * self._drift_freq * self._t + self._drift_phase))
        target = anchor + self._base_offset + drift

        # Keep the resting position fully on screen so words aren't cropped
        target[0] = np.clip(target[0], self._half_w + 4,
                            config.FRAME_WIDTH - self._half_w - 4)
        target[1] = np.clip(target[1], self._half_h + 4,
                            config.FRAME_HEIGHT - self._half_h - 4)

        # ── Spring follow ─────────────────────────────────────────────────────
        if self.pos is None:
            self.pos = target.copy()

        # Apply scatter velocity
        self.pos += self._scatter_vel * dt

        # Spring toward target
        spring = min(config.LABEL_FOLLOW_SPEED * dt, 1.0)
        self.pos += (target - self.pos) * spring

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_dead(self) -> bool:
        """True once the label has fully faded out."""
        return self.fade_state == "out" and self.opacity <= 0.0


# ── Label System ──────────────────────────────────────────────────────────────


class LabelSystem:
    """
    Manages the full collection of floating labels.

    Responsibilities:
      • choose label pool from gender category
      • spawn / remove labels to hit the target count
      • detect body movement and fire scatter events
      • expose current labels to the renderer
    """

    def __init__(self):
        self._labels: list[Label]  = []
        self._category: str        = ""
        self._label_pool: list[str] = []
        self._prev_nose: Optional[tuple] = None
        self._target_count: int    = config.LABEL_COUNT_MID
        self._frame_w: int         = config.FRAME_WIDTH   # updated from landmarks

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        category: str,
        confidence: float,
        landmarks: dict,
        dt: float,
    ):
        person_present = bool(landmarks)

        # ── Movement detection ────────────────────────────────────────────────
        movement_burst = 0.0
        nose = landmarks.get("nose")
        if nose and self._prev_nose:
            mv = float(np.linalg.norm(
                np.array(nose, dtype=float) - np.array(self._prev_nose, dtype=float)
            ))
            if mv > config.MOVEMENT_THRESHOLD:
                movement_burst = mv
        self._prev_nose = nose

        # ── Distance → label count ────────────────────────────────────────────
        if person_present:
            self._target_count = self._estimate_label_count(landmarks)

        # ── Category / pool switch ────────────────────────────────────────────
        if person_present and category != self._category:
            self._category   = category
            self._label_pool = _pool_for(category)
            # Fade out all existing labels so the new pool's labels replace them
            for label in self._labels:
                if not label.is_dead:
                    label.fade_state = "out"

        # ── Anchor map ────────────────────────────────────────────────────────
        anchors = _build_anchors(landmarks)

        # ── Spawn / cull ──────────────────────────────────────────────────────
        # Compute live/hold BEFORE spawn so we can decide how many to add,
        # but update self._labels at the END from self._labels (not live),
        # so newly spawned labels are not immediately discarded.
        hold = sum(1 for l in self._labels if not l.is_dead and l.fade_state != "out")

        if person_present and hold < self._target_count and self._label_pool:
            self._spawn()

        # Fade out excess labels from the front of the list
        if hold > self._target_count:
            excess = hold - self._target_count
            for label in self._labels:
                if not label.is_dead and label.fade_state != "out":
                    label.fade_state = "out"
                    excess -= 1
                    if excess == 0:
                        break

        # ── Update all ────────────────────────────────────────────────────────
        for label in self._labels:
            if not label.is_dead:
                label.update(anchors, dt, movement_burst, person_present)

        # ── Keep words legible — push overlapping labels apart ────────────────
        _separate(self._labels, dt)

        # Remove fully-faded labels (includes newly spawned ones on next tick)
        self._labels = [l for l in self._labels if not l.is_dead]

    def get_labels(self) -> list[Label]:
        return self._labels

    # ── Internal ───────────────────────────────────────────────────────────────

    def _spawn(self):
        # Prefer words not already on screen — keeps the vocabulary varied
        active = {l.text for l in self._labels if not l.is_dead}
        fresh = [t for t in self._label_pool if t not in active]
        text       = random.choice(fresh if fresh else self._label_pool)
        font_size  = random.choice(config.LABEL_FONT_SIZES)
        anchor_key = random.choice(_ANCHOR_SLOTS)
        # Stagger time offset so labels don't start drifting in sync
        time_off   = random.uniform(0.0, 4.0)
        tag = (f"{random.randint(51, 99)}%"
               if random.random() < config.LABEL_TAG_CHANCE else None)
        bracketed  = random.random() < config.LABEL_BRACKET_CHANCE
        glitchy    = self._category == "mixed"
        self._labels.append(Label(
            text, font_size, anchor_key, time_off,
            tag=tag, bracketed=bracketed, glitchy=glitchy,
        ))

    def _estimate_label_count(self, landmarks: dict) -> int:
        ls = landmarks.get("left_shoulder")
        rs = landmarks.get("right_shoulder")
        if ls and rs:
            shoulder_frac = abs(rs[0] - ls[0]) / config.FRAME_WIDTH
            if shoulder_frac > config.NEAR_THRESHOLD:
                return config.LABEL_COUNT_NEAR
            elif shoulder_frac < config.FAR_THRESHOLD:
                return config.LABEL_COUNT_FAR
        return config.LABEL_COUNT_MID


# ── Module-level helpers ───────────────────────────────────────────────────────


def _separate(labels: list[Label], dt: float):
    """
    Soft pairwise repulsion so words don't stack unreadably.

    Each label is treated as an ellipse roughly matching its rendered text
    (wide, short). Overlapping pairs get pushed apart along the line between
    them, with a strength that eases in — the motion stays organic.
    """
    live = [l for l in labels if l.pos is not None and l.opacity > 0.05]
    n = len(live)
    if n < 1:
        return

    push = min(9.0 * dt, 0.6)   # fraction of the overlap corrected per frame
    contain = min(10.0 * dt, 0.7)

    for i in range(n):
        a = live[i]
        aw, ah = a._half_w, a._half_h

        # Soft containment — ease labels back inside the frame
        x, y = float(a.pos[0]), float(a.pos[1])
        if x - aw < 0:
            a.pos[0] += (aw - x) * contain
        elif x + aw > config.FRAME_WIDTH:
            a.pos[0] -= (x + aw - config.FRAME_WIDTH) * contain
        if y - ah < 0:
            a.pos[1] += (ah - y) * contain
        elif y + ah > config.FRAME_HEIGHT:
            a.pos[1] -= (y + ah - config.FRAME_HEIGHT) * contain

        # Keep the specimen readout (bottom-left corner) legible
        rx, ry = 450, config.FRAME_HEIGHT - 150
        if x - aw < rx and y + ah > ry:
            up_cost = (y + ah) - ry
            right_cost = rx - (x - aw)
            if up_cost <= right_cost:
                a.pos[1] -= up_cost * contain
            else:
                a.pos[0] += right_cost * contain

        for j in range(i + 1, n):
            b = live[j]
            bw, bh = b._half_w, b._half_h

            dx = float(b.pos[0] - a.pos[0])
            dy = float(b.pos[1] - a.pos[1])
            ex = (aw + bw) * 0.5
            ey = (ah + bh) * 0.5

            # Normalised elliptical distance: < 1 means visual overlap
            nd = (dx / ex) ** 2 + (dy / ey) ** 2
            if nd >= 1.0 or nd < 1e-6:
                continue

            nd = np.sqrt(nd)
            # Push apart along the offset, weighted by how deep the overlap is
            strength = (1.0 - nd) * push
            ux, uy = dx / (nd + 1e-6), dy / (nd + 1e-6)
            shift = np.array([ux * ex, uy * ey]) * strength
            a.pos -= shift
            b.pos += shift


def _pool_for(category: str) -> list[str]:
    if category == "feminine":
        return list(config.FEMININE_LABELS)
    elif category == "masculine":
        return list(config.MASCULINE_LABELS)
    return list(config.MIXED_LABELS)


def _build_anchors(landmarks: dict) -> dict[str, np.ndarray]:
    """
    Convert the raw landmark pixel dict into named float anchor vectors.
    These are the points that labels orbit around.
    """

    def arr(key: str) -> Optional[np.ndarray]:
        pt = landmarks.get(key)
        return np.array(pt, dtype=float) if pt else None

    nose = arr("nose")
    ls   = arr("left_shoulder")
    rs   = arr("right_shoulder")
    lh   = arr("left_hip")
    rh   = arr("right_hip")

    anchors: dict[str, np.ndarray] = {}

    if nose is not None:
        anchors["head"] = nose + np.array([0.0, -35.0])

    if ls is not None:
        anchors["left_shoulder"] = ls
    if rs is not None:
        anchors["right_shoulder"] = rs

    # Torso centre: average of all available shoulder/hip points
    pts = [p for p in (ls, rs, lh, rh) if p is not None]
    if pts:
        anchors["torso"] = np.mean(pts, axis=0)

    # Hip midpoint — spreads labels down the lower body
    hip_pts = [p for p in (lh, rh) if p is not None]
    if hip_pts:
        anchors["hips"] = np.mean(hip_pts, axis=0)

    # Wide anchors: off to the sides at shoulder height
    if ls is not None and rs is not None:
        mid      = (ls + rs) / 2.0
        half_w   = abs(rs[0] - ls[0]) * 0.85
        anchors["left_wide"]   = np.array([mid[0] - half_w, mid[1]])
        anchors["right_wide"]  = np.array([mid[0] + half_w, mid[1]])
        anchors["upper_left"]  = np.array([mid[0] - half_w * 0.55, mid[1] - 65.0])
        anchors["upper_right"] = np.array([mid[0] + half_w * 0.55, mid[1] - 65.0])

    return anchors
