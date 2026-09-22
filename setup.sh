#!/usr/bin/env bash
# Creates a local virtual environment, installs all dependencies,
# and downloads the MediaPipe pose landmarker model.
# Does NOT install anything globally.

set -e

echo "==> Creating virtual environment (.venv) ..."
python3 -m venv .venv

echo "==> Installing Python dependencies ..."
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "==> Downloading MediaPipe pose model (≈5 MB) ..."
mkdir -p models
curl -L \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" \
  -o models/pose_landmarker_lite.task

echo "==> Pre-downloading InsightFace gender model (≈120 MB, cached in ~/.insightface) ..."
.venv/bin/python3 -c "
import os, warnings
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
warnings.filterwarnings('ignore')
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'genderage'])
app.prepare(ctx_id=-1, det_size=(320, 320))
print('Gender model ready.')
"

echo "==> Preparing models for the web version (web/) ..."
curl -L \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
  -o models/blaze_face_short_range.tflite
cp ~/.insightface/models/buffalo_s/genderage.onnx models/genderage.onnx

echo ""
echo "Setup complete!"
echo ""
echo "To run the artwork:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo "Controls:"
echo "  q  —  quit"
echo "  d  —  toggle debug overlay"
echo "  p  —  toggle presentation mode"
