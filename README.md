# Projected Expectations

An interactive digital artwork that uses machine learning to read, categorise, and label the body in real time.

When a person stands in front of the camera, the system segments their silhouette, fills it with a shifting colour mask, classifies their gender presentation using body proportions, and surrounds them with floating text labels drawn from pools of gendered social expectations.

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

**Colours** — edit `MASK_BASE_HUE` (0–360°) and `LABEL_COLORS` (BGR tuples) to change the colour palette.

**Label density** — edit `LABEL_COUNT_NEAR`, `LABEL_COUNT_MID`, and `LABEL_COUNT_FAR` to control how many words appear at different distances from the camera.

**Classification sensitivity** — edit `MASC_THRESHOLD` / `FEM_THRESHOLD` and `GENDER_SMOOTH_FRAMES` to adjust how quickly and confidently the system commits to a category.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `mediapipe` | Pose landmark detection and body segmentation |
| `opencv-python` | Camera input, image processing, rendering |
| `numpy` | Array math for animation and blending |

---

## Note on Gender Classification

The gender classification in this project is an artistic mechanism, not a claim about truth. It uses a crude body-proportion heuristic (shoulder-to-hip width ratio) that is intentionally imprecise. The piece critiques exactly these kinds of automatic bodily readings — the algorithm's errors and certainties are both part of the work.
