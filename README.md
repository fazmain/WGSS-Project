# Projected Expectations

An interactive digital artwork that uses machine learning to read, categorise, and label the body in real time.

When a person stands in front of the camera, the system segments their silhouette and pours it full of colour — rose/violet or cobalt/cyan, the binary coding worn openly, as critique. The room becomes a deep ink field: a calibration grid, a ghost of the real space, film grain, and a scanline that periodically sweeps down and re-reads the scene. Floating labels drawn from pools of gendered social expectations surround the body, typeset like machine detections — stamped condensed capitals, corner brackets, fabricated confidence percentages. A specimen readout in the corner narrates the classification.

When the algorithm cannot resolve a binary reading, the image itself destabilises: the silhouette drifts through the full spectrum, splits into chromatic fringes, and horizontal slices tear loose. The system's failure to categorise is rendered as the system breaking.

The piece is designed to make visible the automatic, algorithmic nature of social projection — the way bodies are instantly read and surrounded by expectations before a word is spoken.

---

## Setup

Requires Python 3.10+.

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a local virtual environment (`.venv/`)
- Install all dependencies
- Download the MediaPipe pose model (~5 MB) into `models/`

Nothing is installed globally.

---

## Run

```bash
source .venv/bin/activate
python main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle debug overlay (landmarks + segmentation mask) |
| `p` | Toggle presentation mode (hides classification HUD) |

---

## Project Structure

```
.
├── main.py               # Entry point and main loop
├── config.py             # All editable settings (labels, colours, thresholds)
├── requirements.txt      # Python dependencies
├── setup.sh              # One-shot setup script
├── models/               # Downloaded model files (not committed)
└── src/
    ├── tracker.py          # MediaPipe pose detection + body segmentation
    ├── gender_classifier.py # Body proportion classifier (shoulder/hip ratio)
    ├── label_system.py     # Animated floating label engine
    └── renderer.py         # Frame compositor (mask, labels, debug overlay)
```

---

## Customisation

All settings live in `config.py` — no need to touch the source files.

**Label pools** — edit `FEMININE_LABELS`, `MASCULINE_LABELS`, and `MIXED_LABELS` to change what words appear.

**Palettes** — edit `PALETTES` (hex colours per category: body gradient, background ink, accent) to change the colour world.

**Atmosphere** — `BODY_GLOW_STRENGTH`, `BG_GRID_ALPHA`, `SCANLINE_PERIOD`, `BG_GRAIN` and friends control the glow, grid, scanline and grain.

**Glitch** — `GLITCH_SLICE_CHANCE`, `CHROMA_SHIFT_PX` control how hard the image breaks on an unresolved reading.

**Label density** — edit `LABEL_COUNT_NEAR`, `LABEL_COUNT_MID`, and `LABEL_COUNT_FAR` to control how many words appear at different distances from the camera.

**Typography** — `LABEL_FONT_CANDIDATES` / `MONO_FONT_CANDIDATES` (font files), `LABEL_FONT_SIZES`, `LABEL_TAG_CHANCE`, `LABEL_BRACKET_CHANCE`.

**Performance** — `RENDER_SCALE` (colour fields are computed at this fraction of frame size; lower it on slower machines), `CLASSIFY_EVERY` (run the face classifier every Nth frame).

**Classification sensitivity** — edit `MASC_THRESHOLD` / `FEM_THRESHOLD` and `GENDER_SMOOTH_FRAMES` to adjust how quickly and confidently the system commits to a category.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mediapipe` | Pose landmark detection and body segmentation |
| `opencv-python` | Camera input, image processing, rendering |
| `numpy` | Array math for animation and blending |
| `insightface` | Face detection + gender classification |
| `Pillow` | Typography (system fonts, alpha-blended text) |

---

## Note on Gender Classification

The gender classification in this project is an artistic mechanism, not a claim about truth. It uses a crude body-proportion heuristic (shoulder-to-hip width ratio) that is intentionally imprecise. The piece critiques exactly these kinds of automatic bodily readings — the algorithm's errors and certainties are both part of the work.
