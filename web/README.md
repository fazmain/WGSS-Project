# Projected Expectations — web version (sketch)

A browser rebuild of the piece in a risograph-print language: off-white paper,
a yellow halftone body, and the projected labels in fluorescent pink / blue ink
(an unresolved reading overprints both). Everything runs locally in the
browser — no video leaves the machine.

## Run

Requires the models from `../setup.sh` (`models/pose_landmarker_lite.task`,
`models/blaze_face_short_range.tflite`, `models/genderage.onnx`).

```bash
python3 web/serve.py     # serves with caching off, so reloads pick up edits
```

Open <http://localhost:8765/web/> in Chrome and allow the camera.
No camera? <http://localhost:8765/web/?demo> runs a procedural figure.

MediaPipe and ONNX Runtime load from the jsDelivr CDN, so the first load
needs internet.

## Modes

| Key | Mode | Idea |
|-----|------|------|
| `1` | **FILL** | The body is made of words — label rows slide through the silhouette. A feminine reading sets them thin and light, a masculine one wide and heavy. |
| `2` | **PERFORM** | Labels orbit the body and act out the expectation as they age: squeezing narrow and shrinking ("take up less space"), or swelling wide and heavy. Unresolved readings can't hold one form. |
| `3` | **HOLD** | Stand still and letters fly in and stick to you, word by word. Move and they shake loose and fall. |

## Keys

| Key | Action |
|-----|--------|
| `1` `2` `3` | Switch mode |
| `F` `M` `X` | Force a reading (feminine / masculine / unresolved) |
| `A` | Back to the automatic reading |
| `H` | Hide / show captions |
| `D` | Debug overlay (mask, landmarks, fps, motion) |
| double-click | Fullscreen |

URL options: `?demo`, `&mode=1|2|3`, `&cat=f|m|x`, `&idle`, `&still`, `&nohud`.

## How the reading works

Same model as the Python version. Pose landmarks locate the head, a zoomed crop
goes through MediaPipe's face detector, and the face is cropped the way
InsightFace does it and classified by `genderage.onnx` (via onnxruntime-web).
Scores are smoothed over a rolling window. The window resets once nobody has
been in frame for 2 s, so the next visitor starts fresh.

## Files

```
web/
├── index.html
├── fonts/RobotoFlex.woff2   variable font (OFL) — width + weight axes
└── js/
    ├── main.js     render loop, plates, captions, keys
    ├── vision.js   camera, pose + mask, face → gender, demo figure
    ├── modes.js    FILL / PERFORM / HOLD + idle title
    ├── riso.js     paper, inks, misregistration, halftone, variable type
    └── pools.js    label pools (mirrors config.py)
```
