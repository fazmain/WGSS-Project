"""
config.py — All editable settings for Projected Expectations.

Change anything here without touching the rest of the code.
"""

# ── Window ─────────────────────────────────────────────────────────────────────
WINDOW_TITLE  = "Projected Expectations"
CAMERA_INDEX  = 0          # 0 = built-in webcam; main.py auto-detects on macOS
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# ── Body Mask ──────────────────────────────────────────────────────────────────
# The mask cycles through hues over time for an animated colour-shift effect.
# Hue is in degrees (0–360). Speed controls how fast it cycles.
MASK_BASE_HUE   = 300     # starting hue — 300° ≈ hot pink
MASK_HUE_SPEED  = 25      # degrees per second
MASK_SATURATION = 255     # 0–255 (OpenCV uint8)
MASK_VALUE      = 215     # 0–255 brightness
MASK_ALPHA      = 0.62    # overlay strength (0 = transparent, 1 = fully opaque)

# ── Background Mask ────────────────────────────────────────────────────────────
# The background (non-body region) is tinted a different hue for contrast.
BG_MASK_HUE_OFFSET  = 150   # degrees offset from body hue (150° gives a teal/blue contrast)
BG_MASK_SATURATION  = 220   # saturation of background colour
BG_MASK_VALUE       = 190   # brightness of background colour
BG_MASK_ALPHA       = 1.00  # 1.0 = fully solid background colour (matches body mask)

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

# ── Label Colours (BGR format for OpenCV) ──────────────────────────────────────
# Single bright white for maximum readability against the coloured background masks.
LABEL_COLORS = [
    (255, 255, 255),   # white — single colour for legibility
]

# Larger font sizes for readability against the coloured background
LABEL_FONT_SCALES = [0.90, 1.05, 1.25, 1.50]
LABEL_THICKNESS   = 3   # text stroke width in pixels

# ── Label Dynamics ─────────────────────────────────────────────────────────────
LABEL_FADE_SPEED    = 1.2    # opacity units per second (fade-in speed)
LABEL_DRIFT_SPEED   = 30     # max px/s sinusoidal drift
LABEL_FOLLOW_SPEED  = 3.5    # spring constant — how quickly labels follow anchor
SCATTER_VELOCITY    = 140    # px/s impulse added to labels on movement burst
MOVEMENT_THRESHOLD  = 12     # nose movement in px to trigger scatter

# Number of labels shown at different distances
LABEL_COUNT_NEAR    = 28
LABEL_COUNT_MID     = 20
LABEL_COUNT_FAR     = 12

# Shoulder-width / frame-width thresholds that define near/far
NEAR_THRESHOLD = 0.38
FAR_THRESHOLD  = 0.17

# ── Gender Classification ──────────────────────────────────────────────────────
# The classifier accumulates a rolling score over this many frames before
# committing to a category, which prevents rapid flickering.
GENDER_SMOOTH_FRAMES = 50    # ~1.5 s at 30 fps

# Score thresholds (score ranges from −1 = feminine to +1 = masculine)
MASC_THRESHOLD =  0.16
FEM_THRESHOLD  = -0.16

# Minimum filled-window fraction before committing to a non-mixed category
CONFIDENCE_LOW = 0.30
