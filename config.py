"""
config.py — All editable settings for Projected Expectations.

Change anything here without touching the rest of the code.
"""

# ── Window ─────────────────────────────────────────────────────────────────────
WINDOW_TITLE  = "Projected Expectations"
CAMERA_INDEX  = 0          # 0 = built-in webcam; main.py auto-detects on macOS
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# ── Palettes ───────────────────────────────────────────────────────────────────
# Each category gets a palette. Colours are RGB hex strings.
# The binary pink/blue coding is deliberate — the piece critiques exactly that
# coding, so it wears it openly.
#
#   body_top / body_bottom : vertical gradient poured into the silhouette
#   background             : the deep ink field behind the subject
#   accent                 : rim light, scanline, small machine annotations
PALETTES = {
    "feminine": {
        "body_top":    "#FF6FAE",   # rose
        "body_bottom": "#A94DFF",   # violet
        "background":  "#1A0B18",   # deep plum ink
        "accent":      "#FFA3D0",
    },
    "masculine": {
        "body_top":    "#2F6BFF",   # cobalt
        "body_bottom": "#2FE0E8",   # cyan
        "background":  "#060D1D",   # deep navy ink
        "accent":      "#8FD8FF",
    },
    # "mixed" body colour is dynamic (a slow full-spectrum drift computed in the
    # renderer) — the palette below supplies its background and accent.
    "mixed": {
        "body_top":    "#FFFFFF",   # placeholder — overridden by spectrum drift
        "body_bottom": "#FFFFFF",
        "background":  "#0D0A16",   # violet-charcoal ink
        "accent":      "#D9C8FF",
    },
}

# Seconds for the palette to glide from one category to another
PALETTE_LERP_TIME = 1.4

# Colour fields (gradients, glow, grain) are computed at this fraction of the
# frame size and upscaled — big speedup, no visible loss on soft imagery.
RENDER_SCALE = 0.5

# ── Body Field ─────────────────────────────────────────────────────────────────
BODY_SHIMMER_STRENGTH = 0.07   # amplitude of the silk-like moving bands (0–0.3)
BODY_SHIMMER_PERIOD   = 190    # px wavelength of the bands
BODY_SHIMMER_SPEED    = 0.35   # bands per second drifting upward
BODY_GRAIN            = 7      # film grain amplitude inside the body (0–20)
BODY_GLOW_STRENGTH    = 0.65   # outer halo brightness (0–1)
BODY_RIM_STRENGTH     = 0.60   # thin bright rim on the silhouette edge (0–1)

# "mixed" glitch behaviour — the algorithm failing to hold a reading
GLITCH_SLICE_CHANCE   = 0.030  # per-frame chance of a displaced horizontal slice
GLITCH_MAX_SHIFT      = 26     # max px horizontal displacement of a slice
CHROMA_SHIFT_PX       = 4      # chromatic aberration offset for mixed silhouette

# ── Background Field ───────────────────────────────────────────────────────────
BG_GHOST_ALPHA     = 0.10   # how much of the real room bleeds through (0–0.3)
BG_GRID_SPACING    = 80     # px between calibration grid lines
BG_GRID_ALPHA      = 0.070  # grid line brightness (0–0.3)
BG_VIGNETTE        = 0.42   # darkening at the corners (0–1)
BG_GRAIN           = 5      # film grain amplitude on the background
SCANLINE_PERIOD    = 9.0    # seconds between scanline sweeps
SCANLINE_SPEED     = 260    # px/s downward sweep speed
SCANLINE_STRENGTH  = 0.16   # brightness of the sweeping band

# ── Label Pools ────────────────────────────────────────────────────────────────
# Edit these lists freely — they drive what text appears around the body.

FEMININE_LABELS = [
    "smile more",
    "pretty",
    "too emotional",
    "be modest",
    "likable",
    "soft",
    "quiet",
    "nurturing",
    "polite",
    "desirable",
    "graceful",
    "delicate",
    "fragile",
    "agreeable",
    "pleasing",
    "submissive",
    "dainty",
    "passive",
    "decorative",
    "gentle",
    "sweet",
    "meek",
    "self-sacrifice",
    "tame yourself",
    "take up less space",
    "be agreeable",
    "don't be difficult",
    "be soft",
    "look approachable",
    "stay small",
]

MASCULINE_LABELS = [
    "strong",
    "leader",
    "dominant",
    "confident",
    "aggressive",
    "ambitious",
    "tough",
    "stoic",
    "powerful",
    "provider",
    "assertive",
    "in control",
    "breadwinner",
    "invulnerable",
    "decisive",
    "competitive",
    "unemotional",
    "self-reliant",
    "imposing",
    "disciplined",
    "rational",
    "fearless",
    "commanding",
    "man up",
    "don't cry",
    "take charge",
    "be logical",
    "no weakness",
    "prove yourself",
    "dominate",
]

MIXED_LABELS = [
    "too much",
    "difficult",
    "confusing",
    "not enough",
    "perform",
    "explain yourself",
    "choose",
    "readable",
    "wrong",
    "undefined",
    "ambiguous",
    "illegible",
    "pick a side",
    "make up your mind",
    "you don't fit",
    "either/or",
    "in between",
    "neither",
    "transgressive",
    "unclassified",
    "deviant",
    "outlier",
    "non-conforming",
    "resist categorization",
    "beyond the binary",
]

# ── Typography ─────────────────────────────────────────────────────────────────
# First existing font in each list is used. (path, ttc_index)
LABEL_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", 0),
    ("/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf", 0),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
]
MONO_FONT_CANDIDATES = [
    ("/System/Library/Fonts/Menlo.ttc", 0),
    ("/System/Library/Fonts/SFNSMono.ttf", 0),
    ("/System/Library/Fonts/Supplemental/Courier New.ttf", 0),
]

LABEL_FONT_SIZES   = [30, 38, 48, 62]   # px — one is picked per label
LABEL_TRACKING     = 2.5                # extra px between letters (stamped look)
LABEL_TAG_CHANCE   = 0.55               # fraction of labels that get a % tag
LABEL_BRACKET_CHANCE = 0.30             # fraction drawn as detection boxes

# ── Label Dynamics ─────────────────────────────────────────────────────────────
LABEL_FADE_SPEED    = 1.2    # opacity units per second (fade-in speed)
LABEL_DRIFT_SPEED   = 30     # max px/s sinusoidal drift
LABEL_FOLLOW_SPEED  = 3.5    # spring constant — how quickly labels follow anchor
SCATTER_VELOCITY    = 140    # px/s impulse added to labels on movement burst
MOVEMENT_THRESHOLD  = 12     # nose movement in px to trigger scatter

# Number of labels shown at different distances
LABEL_COUNT_NEAR    = 26
LABEL_COUNT_MID     = 18
LABEL_COUNT_FAR     = 11

# Shoulder-width / frame-width thresholds that define near/far
NEAR_THRESHOLD = 0.38
FAR_THRESHOLD  = 0.17

# ── Classifier cadence ─────────────────────────────────────────────────────────
# Run the (CPU-heavy) face classifier every Nth frame; the rolling window
# smoothing makes the skipped frames invisible.
CLASSIFY_EVERY = 3

# ── Gender Classification ──────────────────────────────────────────────────────
# The classifier accumulates a rolling score over this many frames before
# committing to a category, which prevents rapid flickering.
GENDER_SMOOTH_FRAMES = 50    # ~1.5 s at 30 fps

# Score thresholds (score ranges from −1 = feminine to +1 = masculine)
MASC_THRESHOLD =  0.16
FEM_THRESHOLD  = -0.16

# Minimum filled-window fraction before committing to a non-mixed category
CONFIDENCE_LOW = 0.30
