// vision.js — camera, body tracking, segmentation, and the gender "reading".
//
// Camera path : MediaPipe PoseLandmarker (landmarks + segmentation mask)
//               MediaPipe FaceDetector → 96×96 crop → InsightFace genderage.onnx
//               (the same model the Python version uses) via onnxruntime-web.
// Demo path   : a procedural figure, so the piece can be viewed / tested
//               without a camera (open with ?demo).
//
// Everything is exposed in *display* orientation (already mirrored), with
// landmarks normalised to [0,1] of the video frame.

import { clamp } from "./riso.js";

const MP_VER = "1.0.1";
const MP_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const POSE_MODEL = "../models/pose_landmarker_lite.task";
const FACE_MODEL = "../models/blaze_face_short_range.tflite";
const GENDER_MODEL = "../models/genderage.onnx";

// Classification smoothing (mirrors config.py)
const WINDOW = 24;          // gender samples in the rolling window
const CLASSIFY_EVERY = 5;   // frames between face classifications
const MASC_T = 0.16, FEM_T = -0.16, CONF_LOW = 0.3;
const FORGET_AFTER_MS = 2000;   // absent this long → forget the last person

export const LM = {
  nose: 0, lEye: 2, rEye: 5, lEar: 7, rEar: 8,
  lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16,
  lHip: 23, rHip: 24, lKnee: 25, rKnee: 26,
};

export class Vision {
  constructor() {
    this.MW = 480; this.MH = 270;
    this.aspect = 16 / 9;
    this.landmarks = null;
    this.present = false;
    this.lastSeen = -1e9;
    this.motion = 0;          // 0–1, smoothed movement energy
    this.rawMotion = 0;
    this.lmSpeed = 0;
    this._lmTime = 0;
    this.scores = [];
    this.category = "mixed";
    this.confidence = 0;
    this.override = null;     // forced category (keys F / M / X)
    this.faceBox = null;
    this.status = "starting";
    this.frame = 0;
    this._lastTs = 0;
    this._busy = false;
  }

  // ── Setup ────────────────────────────────────────────────────────────────────

  async init({ demo = false, idle = false, still = false, demoCat = null } = {}) {
    this.demo = demo;
    if (demo) {
      this._alloc(480, 270);
      this.demoBody = new DemoBody({ idle, still });
      this.demoCat = demoCat;
      this.status = "demo";
      return;
    }

    this.status = "requesting camera";
    const video = document.createElement("video");
    video.muted = true; video.playsInline = true;
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    await video.play();
    this.video = video;
    this.aspect = video.videoWidth / video.videoHeight || 16 / 9;
    this._alloc(480, Math.round(480 / this.aspect));

    // Small copy of the frame for the pose model — keeps the mask cheap to read
    this.proc = document.createElement("canvas");
    this.proc.width = this.MW; this.proc.height = this.MH;
    this.pctx = this.proc.getContext("2d", { willReadFrequently: false });

    this.status = "loading pose model";
    const { FilesetResolver, PoseLandmarker, FaceDetector } = await import(`${MP_URL}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MP_URL}/wasm`);
    const makePose = delegate => PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate },
      runningMode: "VIDEO",
      numPoses: 1,
      outputSegmentationMasks: true,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    // CPU on purpose: with the GPU delegate the segmentation mask lives in a
    // float texture, and reading it back (getAsFloat32Array) fails on macOS
    // Chrome ("glReadPixels: invalid format and type") — the mask comes back
    // all zeros and the silhouette vanishes. The CPU mask needs no readback.
    this.pose = await makePose("CPU");

    this.status = "loading face models";
    try {
      // Run on a zoomed crop around the head (found via pose landmarks): the
      // short-range detector only sees faces that fill a good part of its input.
      this.face = await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: FACE_MODEL, delegate: "CPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: 0.5,
      });
      this.head = document.createElement("canvas");
      this.head.width = this.head.height = 256;
      this.hctx = this.head.getContext("2d");
      const ort = await import(`${ORT_URL}ort.wasm.min.mjs`);
      ort.env.wasm.wasmPaths = ORT_URL;
      ort.env.wasm.numThreads = 1;
      this.ort = ort;
      this.gender = await ort.InferenceSession.create(GENDER_MODEL, { executionProviders: ["wasm"] });
      this.crop = document.createElement("canvas");
      this.crop.width = this.crop.height = 96;
      this.cctx = this.crop.getContext("2d", { willReadFrequently: true });
    } catch (e) {
      // The piece still runs; category can be forced with F / M / X.
      console.warn("Gender classifier unavailable:", e);
      this.gender = null;
    }
    this.status = "live";
  }

  _alloc(w, h) {
    this.MW = w; this.MH = h;
    const n = w * h;
    this.mask = new Float32Array(n);   // smoothed person mask, display orientation
    this.core = new Float32Array(n);   // blurred mask — "depth" for halftone shading
    this._tmp = new Float32Array(n);
    this.maskCanvas = document.createElement("canvas");
    this.maskCanvas.width = w; this.maskCanvas.height = h;
    this._mctx = this.maskCanvas.getContext("2d");
    this._mimg = this._mctx.createImageData(w, h);
  }

  // ── Per-frame ────────────────────────────────────────────────────────────────

  update(now, dt) {
    this.frame++;
    if (this.demo) {
      const out = this.demoBody.step(now / 1000, this.MW, this.MH);
      this.landmarks = out.landmarks;
      if (out.landmarks) { this.lastSeen = now; this._trackSpeed(now); }
      this._ingest(out.mask, false, 1, 1);
    } else if (this.status === "live" && this.video.readyState >= 2 &&
               this.video.currentTime !== this._lastVideoTime) {
      this._lastVideoTime = this.video.currentTime;
      this.pctx.drawImage(this.video, 0, 0, this.MW, this.MH);
      const ts = Math.max(this._lastTs + 1, Math.round(now));
      this._lastTs = ts;
      this.pose.detectForVideo(this.proc, ts, res => {
        const lms = res.landmarks && res.landmarks[0];
        const m = res.segmentationMasks && res.segmentationMasks[0];
        if (lms) {
          this.landmarks = lms.map(l => ({ x: 1 - l.x, y: l.y, v: l.visibility ?? 1 }));
          this.lastSeen = now;
          this._trackSpeed(now);
        } else {
          this.landmarks = null;
        }
        if (m && lms) this._ingest(m.getAsFloat32Array(), true, m.width / this.MW, m.height / this.MH);
        else this._ingest(null);
      });
      if (this.gender && this.present && !this._busy && this.frame % CLASSIFY_EVERY === 0) {
        this._classify();
      }
    }

    this.present = now - this.lastSeen < 600;
    if (now - this.lastSeen > FORGET_AFTER_MS) this.scores.length = 0;

    // Movement energy: how much the silhouette changed, or how fast the
    // hands / head are moving — whichever is stronger.
    if (now - this._lmTime > 250) this.lmSpeed *= 0.9;
    const eMask = clamp((this.rawMotion - 0.02) / 0.10, 0, 1);
    const eLm = clamp((this.lmSpeed - 0.6) / 1.8, 0, 1);
    const e = Math.max(eMask, eLm);
    this.motion += (e - this.motion) * (e > this.motion ? 0.35 : 0.05);
    if (!this.present) this.motion *= 0.9;

    this._categorise(now);
  }

  /** Speed of the fastest limbs, in shoulder-widths per second. */
  _trackSpeed(now) {
    const l = this.landmarks, p = this._prevLm;
    const dt = (now - (this._lmTime || now)) / 1000;
    this._prevLm = l; this._lmTime = now;
    if (!p || dt <= 0 || dt > 0.25) return;
    const a = this.aspect;
    const ls = l[LM.lSh], rs = l[LM.rSh];
    if (ls.v < 0.35 || rs.v < 0.35) return;
    const sw = Math.hypot((ls.x - rs.x) * a, ls.y - rs.y) || 0.1;
    const speeds = [];
    for (const i of [LM.nose, LM.lWr, LM.rWr, LM.lEl, LM.rEl, LM.lSh, LM.rSh]) {
      if (l[i].v < 0.35 || p[i].v < 0.35) continue;
      speeds.push(Math.hypot((l[i].x - p[i].x) * a, l[i].y - p[i].y) / dt / sw);
    }
    if (!speeds.length) return;
    speeds.sort((x, y) => y - x);
    const top = (speeds[0] + (speeds[1] ?? speeds[0])) / 2;
    this.lmSpeed += (top - this.lmSpeed) * 0.5;
  }

  _ingest(src, mirror = false, sx = 1, sy = 1) {
    const { mask, MW, MH } = this;
    let diff = 0, sum = 0;
    let x0 = MW, y0 = MH, x1 = -1, y1 = -1;
    const srcW = Math.round(MW * sx);
    for (let y = 0; y < MH; y++) {
      const row = y * MW;
      const srow = Math.min(Math.round(y * sy), Math.round(MH * sy) - 1) * srcW;
      for (let x = 0; x < MW; x++) {
        const i = row + x;
        let s = 0;
        if (src) {
          const xs = Math.min(Math.round((mirror ? MW - 1 - x : x) * sx), srcW - 1);
          s = src[srow + xs];
        }
        const old = mask[i];
        const nv = old + (s - old) * 0.55;   // temporal smoothing kills flicker
        mask[i] = nv;
        diff += Math.abs(nv - old);
        sum += nv;
        if (nv > 0.3) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          y1 = y;
        }
      }
    }
    this.rawMotion = sum > MW * MH * 0.004 ? diff / sum : 0;
    this.maskArea = sum / (MW * MH);
    // Bounding box of the body in mask pixels (null when nobody is there)
    this.bounds = x1 >= 0 ? [x0, y0, x1 + 1, y1 + 1] : null;

    boxBlur(mask, this.core, this._tmp, MW, MH, Math.round(MW * 0.022));

    // Alpha image of the mask — used to clip text to the body
    const d = this._mimg.data;
    for (let i = 0, n = MW * MH; i < n; i++) {
      const m = mask[i];
      const a = m <= 0.3 ? 0 : m >= 0.62 ? 255 : ((m - 0.3) / 0.32) * 255;
      d[i * 4 + 3] = a;
    }
    this._mctx.putImageData(this._mimg, 0, 0);
  }

  /** Head centre + size in video pixels, from the pose landmarks. */
  _headHint(vw, vh) {
    const l = this.landmarks;
    if (!l) return null;
    const P = (i, min = 0.35) => (l[i].v >= min ? [(1 - l[i].x) * vw, l[i].y * vh] : null);
    const nose = P(LM.nose), le = P(LM.lEye), re = P(LM.rEye);
    const lear = P(LM.lEar, 0.2), rear = P(LM.rEar, 0.2), ls = P(LM.lSh), rs = P(LM.rSh);
    const c = nose || (le && re ? [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2] : null);
    if (!c) return null;
    const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const size = lear && rear ? d(lear, rear) : le && re ? d(le, re) * 2.4 : ls && rs ? d(ls, rs) * 0.45 : null;
    return size ? { x: c[0], y: c[1], size } : null;
  }

  async _classify() {
    this._busy = true;
    try {
      const vw = this.video.videoWidth, vh = this.video.videoHeight;
      const hint = this._headHint(vw, vh);
      if (!hint) return;
      // Zoom on the head, find the face there
      const S = hint.size * 3.2, hx0 = hint.x - S / 2, hy0 = hint.y - S / 2;
      this.hctx.fillStyle = "#000";
      this.hctx.fillRect(0, 0, 256, 256);
      this.hctx.drawImage(this.video, hx0, hy0, S, S, 0, 0, 256, 256);
      const dets = (this.face.detect(this.head).detections || []).filter(d => d.boundingBox);
      if (!dets.length) { this.faceBox = null; return; }
      const dist = d => Math.hypot(d.boundingBox.originX + d.boundingBox.width / 2 - 128,
                                   d.boundingBox.originY + d.boundingBox.height / 2 - 128);
      const fb = dets.reduce((a, c) => (dist(c) < dist(a) ? c : a)).boundingBox;
      const k = S / 256;
      const b = { originX: hx0 + fb.originX * k, originY: hy0 + fb.originY * k, width: fb.width * k, height: fb.height * k };
      // InsightFace genderage preprocessing: square crop 1.5× the face box,
      // resized to 96×96, RGB, raw 0–255 values, NCHW. BlazeFace boxes sit a
      // little tighter and lower than InsightFace's detector, so compensate.
      const cx = b.originX + b.width / 2, cy = b.originY + b.height * 0.42;
      const side = Math.max(b.width, b.height) * 1.5 * 1.2;
      const g = this.cctx;
      g.fillStyle = "#000"; g.fillRect(0, 0, 96, 96);
      g.drawImage(this.video, cx - side / 2, cy - side / 2, side, side, 0, 0, 96, 96);
      const px = g.getImageData(0, 0, 96, 96).data;
      const N = 96 * 96, data = new Float32Array(3 * N);
      for (let i = 0; i < N; i++) {
        data[i] = px[i * 4]; data[N + i] = px[i * 4 + 1]; data[2 * N + i] = px[i * 4 + 2];
      }
      const input = new this.ort.Tensor("float32", data, [1, 3, 96, 96]);
      const out = await this.gender.run({ [this.gender.inputNames[0]]: input });
      const p = out[this.gender.outputNames[0]].data;   // [female, male, age/100]
      this.scores.push(p[1] > p[0] ? 1 : -1);
      if (this.scores.length > WINDOW) this.scores.shift();
      this.faceBox = { x: 1 - (b.originX + b.width) / vw, y: b.originY / vh, w: b.width / vw, h: b.height / vh };
    } catch (e) {
      console.warn("classify failed:", e);
    } finally {
      this._busy = false;
    }
  }

  _categorise(now) {
    if (this.override) { this.category = this.override; this.confidence = 1; return; }
    if (this.demo) {
      const cats = ["feminine", "masculine", "mixed"];
      this.category = this.demoCat || cats[Math.floor(now / 12000) % 3];
      this.confidence = 0.62 + 0.3 * Math.abs(Math.sin(now / 3100));
      return;
    }
    const n = this.scores.length;
    const conf = n / WINDOW;
    this.confidence = conf;
    if (!n || conf < CONF_LOW) { this.category = "mixed"; return; }
    const mean = this.scores.reduce((a, b) => a + b, 0) / n;
    this.category = mean > MASC_T ? "masculine" : mean < FEM_T ? "feminine" : "mixed";
  }

  // ── Queries ──────────────────────────────────────────────────────────────────

  /** Shoulder width as a fraction of frame width → 0 (far) … 1 (near). */
  closeness() {
    const l = this.landmarks;
    if (!l) return 0.4;
    const a = l[LM.lSh], b = l[LM.rSh];
    if (!a || !b || a.v < 0.35 || b.v < 0.35) return 0.4;
    return clamp((Math.abs(a.x - b.x) - 0.12) / 0.26, 0, 1);
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function boxBlur(src, dst, tmp, w, h, r) {
  const inv = 1 / (2 * r + 1);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let acc = 0;
    for (let x = -r; x <= r; x++) acc += src[row + clamp(x, 0, w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc * inv;
      acc += src[row + Math.min(x + r + 1, w - 1)] - src[row + Math.max(x - r, 0)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -r; y <= r; y++) acc += tmp[clamp(y, 0, h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = acc * inv;
      acc += tmp[Math.min(y + r + 1, h - 1) * w + x] - tmp[Math.max(y - r, 0) * w + x];
    }
  }
}

// ── Demo figure ────────────────────────────────────────────────────────────────
// A soft procedural body that sways, drifts nearer/farther and periodically
// waves its arms — enough to exercise every mode without a camera.

class DemoBody {
  constructor({ idle, still }) {
    this.idle = idle; this.still = still;
    this.c = document.createElement("canvas");
    this.g = null;
  }

  step(t, W, H) {
    if (this.idle) return { landmarks: null, mask: null };
    if (!this.g || this.c.width !== W) {
      this.c.width = W; this.c.height = H;
      this.g = this.c.getContext("2d", { willReadFrequently: true });
    }
    const g = this.g;
    const k = 1.0 + 0.2 * Math.sin(t * 0.11);
    const U = H * k;
    const C = [W * (0.5 + 0.09 * Math.sin(t * 0.33)), H * 0.58 + Math.sin(t * 1.1) * 2];
    const tilt = 0.05 * Math.sin(t * 0.5);

    // Wave: every 11 s the figure throws its arms up for ~2.6 s
    const cyc = t % 11;
    const wave = this.still ? 0 : cyc > 6 && cyc < 8.6 ? Math.sin(((cyc - 6) / 2.6) * Math.PI) : 0;
    const wig = Math.sin(t * 9) * 0.09 * wave;

    const rot = ([x, y]) => {
      const c = Math.cos(tilt), s = Math.sin(tilt);
      return [C[0] + (x * c - y * s) * U, C[1] + (x * s + y * c) * U];
    };
    const mix = (a, b, m) => [a[0] + (b[0] - a[0]) * m, a[1] + (b[1] - a[1]) * m];
    const P = {
      head:  rot([0, -0.31]),
      neck:  rot([0, -0.19]),
      lSh:   rot([-0.17, -0.13]), rSh: rot([0.17, -0.13]),
      lEl:   rot(mix([-0.23, 0.07], [-0.30, -0.22], wave)),
      rEl:   rot(mix([0.23, 0.07], [0.30, -0.22], wave)),
      lWr:   rot(mix([-0.22, 0.28], [-0.34 + wig, -0.46], wave)),
      rWr:   rot(mix([0.22, 0.28], [0.34 - wig, -0.46], wave)),
      lHip:  rot([-0.11, 0.27]), rHip: rot([0.11, 0.27]),
      lKnee: rot([-0.12, 0.62]), rKnee: rot([0.12, 0.62]),
    };

    g.clearRect(0, 0, W, H);
    g.filter = "blur(1.2px)";
    g.fillStyle = g.strokeStyle = "#fff";
    g.lineCap = g.lineJoin = "round";
    // torso
    g.beginPath();
    for (const p of [[-0.19, -0.13], [0.19, -0.13], [0.14, 0.12], [0.16, 0.32], [0.15, 0.9], [-0.15, 0.9], [-0.16, 0.32], [-0.14, 0.12]].map(rot)) g.lineTo(p[0], p[1]);
    g.closePath(); g.fill();
    // neck + head
    g.lineWidth = 0.1 * U;
    g.beginPath(); g.moveTo(...P.neck); g.lineTo(...P.head); g.stroke();
    g.beginPath(); g.ellipse(P.head[0], P.head[1], 0.075 * U, 0.092 * U, tilt, 0, Math.PI * 2); g.fill();
    // arms
    g.lineWidth = 0.085 * U;
    for (const [a, b, c] of [[P.lSh, P.lEl, P.lWr], [P.rSh, P.rEl, P.rWr]]) {
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.lineTo(...c); g.stroke();
    }
    // legs
    g.lineWidth = 0.15 * U;
    for (const [a, b] of [[P.lHip, P.lKnee], [P.rHip, P.rKnee]]) {
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
    }
    g.filter = "none";

    const px = g.getImageData(0, 0, W, H).data;
    const mask = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) mask[i] = px[i * 4 + 3] / 255;

    const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, v: 0 }));
    const put = (i, p) => { lm[i] = { x: p[0] / W, y: p[1] / H, v: 1 }; };
    put(LM.nose, [P.head[0], P.head[1] + 0.01 * U]);
    put(LM.lEye, [P.head[0] - 0.03 * U, P.head[1] - 0.02 * U]);
    put(LM.rEye, [P.head[0] + 0.03 * U, P.head[1] - 0.02 * U]);
    for (const key of ["lSh", "rSh", "lEl", "rEl", "lWr", "rWr", "lHip", "rHip", "lKnee", "rKnee"]) put(LM[key], P[key]);
    return { landmarks: lm, mask };
  }
}
