// main.js — boot, render loop, keys, HUD.
//
// Keys:  1 FILL · 2 PERFORM · 3 HOLD
//        F / M / X force a reading (feminine / masculine / unresolved), A = auto
//        H hide/show captions · D debug · double-click fullscreen
// URL:   ?demo (no camera) &mode=1|2|3 &cat=f|m|x &idle &still &nohud

import {
  PAPER, INK, clamp, lerp, loadFonts, makePaperTile, makeSpeckleTile,
  halftone, inkOffset, drawText,
} from "./riso.js";
import { Vision } from "./vision.js";
import { Fill, Perform, Hold, anchorsFrom, drawIdle } from "./modes.js";

const params = new URLSearchParams(location.search);
const CAT_KEYS = { f: "feminine", m: "masculine", x: "mixed" };
const MODE_NAMES = ["", "FILL", "PERFORM", "HOLD"];

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");
// Offscreen plates, like separate riso drums:
const layer = document.createElement("canvas");   // scratch plate for clipping
const lctx = layer.getContext("2d");
const bodyPlate = document.createElement("canvas");   // yellow halftone body
const bctx = bodyPlate.getContext("2d");
const inkPlate = document.createElement("canvas");    // pink / blue projection
const ictx = inkPlate.getContext("2d");
const PLATES = [layer, bodyPlate, inkPlate];

function clearPlate(c) {
  c.save();
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.globalCompositeOperation = "source-over";
  c.globalAlpha = 1;
  c.clearRect(0, 0, c.canvas.width, c.canvas.height);
  c.restore();
}

let W = 0, H = 0, DPR = 1, u = 1;
let paperPat, specklePat;
let mode = clamp(+params.get("mode") || 1, 1, 3);
let showHud = !params.has("nohud");
let debug = false;
let error = null;
let idleAlpha = 1;
let prevMotion = 0;
let fps = 60;

const vision = new Vision();
const modes = { 1: new Fill(), 2: new Perform(), 3: new Hold() };
window.__vision = vision;   // handy in the devtools console
window.__modes = modes;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth; H = window.innerHeight;
  u = Math.min(W / 1440, H / 900);
  for (const c of [canvas, ...PLATES]) { c.width = Math.round(W * DPR); c.height = Math.round(H * DPR); }
  for (const c of [lctx, bctx, ictx]) c.setTransform(DPR, 0, 0, DPR, 0, 0);
  const crisp = new DOMMatrix().scale(1 / DPR);
  paperPat = ctx.createPattern(paperTile, "repeat"); paperPat.setTransform(crisp);
  specklePat = ctx.createPattern(speckleTile, "repeat"); specklePat.setTransform(crisp);
}

// ── Frame ──────────────────────────────────────────────────────────────────────

let last = performance.now();
function loop(now) {
  const dt = clamp((now - last) / 1000, 0, 0.1);
  last = now;
  fps = lerp(fps, 1 / Math.max(dt, 1e-3), 0.05);
  const t = now / 1000;

  try { vision.update(now, dt); } catch (e) { error = error || String(e); console.error(e); }

  // Cover-fit the (mirrored) video frame to the screen
  const va = vision.aspect;
  const vw = W / H > va ? W : H * va;
  const vh = W / H > va ? W / va : H;
  const ox = (W - vw) / 2, oy = (H - vh) / 2;
  const { MW, MH, mask, core } = vision;
  const toScreen = (x, y) => [ox + x * vw, oy + y * vh];
  const sampleAt = (arr, x, y) => {
    const mx = ((x - ox) / vw) * MW - 0.5, my = ((y - oy) / vh) * MH - 0.5;
    if (mx < 0 || my < 0 || mx >= MW - 1 || my >= MH - 1) return 0;
    const ix = mx | 0, iy = my | 0, fx = mx - ix, fy = my - iy, i = iy * MW + ix;
    const a = arr[i] + (arr[i + 1] - arr[i]) * fx;
    const b = arr[i + MW] + (arr[i + MW + 1] - arr[i + MW]) * fx;
    return a + (b - a) * fy;
  };
  const b = vision.bounds;
  const pad = 12;
  const bounds = b ? [ox + (b[0] / MW) * vw - pad, oy + (b[1] / MH) * vh - pad,
                      ox + (b[2] / MW) * vw + pad, oy + (b[3] / MH) * vh + pad] : null;

  const motion = vision.motion;
  const spike = motion - prevMotion > 0.1 ? motion : 0;
  prevMotion = motion;

  const F = {
    ctx: ictx, lctx, layer, W, H, u, t, dt,
    cat: vision.category, present: vision.present,
    motion, spike, closeness: vision.closeness(),
    anchors: anchorsFrom(vision.landmarks, toScreen, H),
    mask: (x, y) => sampleAt(mask, x, y),
    maskCanvas: vision.maskCanvas, maskRect: [ox, oy, vw, vh], bounds,
  };

  clearPlate(bctx);
  clearPlate(ictx);

  // Body plate: a yellow halftone, shaded by how deep inside the silhouette a
  // point lies (the blurred mask).
  if (bounds) {
    bctx.fillStyle = INK.yellow;
    const o = inkOffset("yellow", t, motion);
    bctx.save();
    bctx.translate(o[0], o[1]);
    halftone(bctx, W, H, (x, y) => {
      const m = sampleAt(mask, x, y);
      if (m < 0.05) return 0;
      const c = sampleAt(core, x, y);
      return 0.85 * m * (0.28 + 0.72 * c * c);
    }, { spacing: Math.max(6, 8.5 * u), angle: 75, gamma: 0.7, bounds });
    bctx.restore();
  }

  // Ink plate: the projection
  try { modes[mode].frame(F); } catch (e) { error = error || String(e); console.error(e); }

  // Knockout: no yellow prints under the pink/blue, so the binary coding
  // stays legible (pink and blue still overprint each other → purple).
  bctx.globalCompositeOperation = "destination-out";
  bctx.drawImage(inkPlate, 0, 0, W, H);
  bctx.globalCompositeOperation = "source-over";

  // Print: paper, then each plate multiplied on top
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 1;
  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "multiply";
  ctx.drawImage(bodyPlate, 0, 0, W, H);
  ctx.drawImage(inkPlate, 0, 0, W, H);
  F.ctx = ctx;

  // Idle title once nobody has been in frame for a moment
  const idleTarget = vision.present ? 0 : 1;
  idleAlpha += (idleTarget - idleAlpha) * Math.min(1, dt * (idleTarget ? 1.1 : 4));
  if (idleAlpha > 0.01) drawIdle(F, idleAlpha);

  // Print texture: ink voids, then paper fibre
  ctx.globalCompositeOperation = "source-over";
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = specklePat;
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = 1;
  ctx.fillStyle = paperPat;
  ctx.fillRect(0, 0, W, H);

  if (showHud) drawHud(F);
  if (debug) drawDebug(F, toScreen);

  requestAnimationFrame(loop);
}

// ── Captions ───────────────────────────────────────────────────────────────────

function drawHud(F) {
  const s = Math.max(11, 13 * u);
  const m = 28 * u;
  ctx.globalCompositeOperation = "multiply";
  ctx.fillStyle = INK.black;
  ctx.textBaseline = "alphabetic";

  ctx.letterSpacing = `${(2.5 * u).toFixed(1)}px`;
  drawText(ctx, "PROJECTED EXPECTATIONS", m, m + s, s, 800, 125, "left");
  ctx.letterSpacing = `${(1 * u).toFixed(1)}px`;
  drawText(ctx, `Nº 0${mode} — ${MODE_NAMES[mode]}`, m, m + s * 2.5, s, 380, 100, "left");

  let status;
  if (error) status = `ERROR — ${error}`;
  else if (vision.status !== "live" && vision.status !== "demo") status = vision.status.toUpperCase() + "…";
  else if (!vision.present) status = "AWAITING SUBJECT";
  else {
    const c = vision.category === "mixed" ? "UNRESOLVED" : vision.category.toUpperCase();
    const forced = vision.override ? "  (FORCED)" : "";
    status = `READ AS — ${c}   ${Math.round(vision.confidence * 100)}%${forced}`;
  }
  ctx.letterSpacing = `${(2 * u).toFixed(1)}px`;
  drawText(ctx, status, m, H - m, s, 560, 110, "left");

  ctx.letterSpacing = `${(1.5 * u).toFixed(1)}px`;
  const keys = "1 FILL   2 PERFORM   3 HOLD   ·   F M X A READING   ·   H HIDE";
  drawText(ctx, keys, W - m, H - m, Math.max(10, 11 * u), 420, 100, "right");
  ctx.letterSpacing = "0px";
}

function drawDebug(F, toScreen) {
  ctx.globalCompositeOperation = "source-over";
  const tw = 240, th = tw / vision.aspect;
  ctx.fillStyle = "#000";
  ctx.fillRect(W - tw - 20, 20, tw, th);
  ctx.save();
  ctx.globalCompositeOperation = "destination-out";
  ctx.drawImage(vision.maskCanvas, W - tw - 20, 20, tw, th);
  ctx.restore();
  ctx.strokeStyle = "#0a0"; ctx.strokeRect(W - tw - 20, 20, tw, th);
  if (vision.landmarks) {
    ctx.fillStyle = "#0a0";
    for (const l of vision.landmarks) {
      if (l.v < 0.35) continue;
      const [x, y] = toScreen(l.x, l.y);
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
    }
  }
  if (vision.faceBox) {
    const f = vision.faceBox, [x, y] = toScreen(f.x, f.y), [x2, y2] = toScreen(f.x + f.w, f.y + f.h);
    ctx.strokeStyle = "#f0f"; ctx.strokeRect(x, y, x2 - x, y2 - y);
  }
  ctx.font = "12px monospace"; ctx.fillStyle = "#0a0";
  const lines = [
    `fps ${fps.toFixed(0)}   status ${vision.status}`,
    `present ${vision.present}   motion ${vision.motion.toFixed(2)} (raw ${vision.rawMotion.toFixed(3)})`,
    `closeness ${F.closeness.toFixed(2)}   cat ${vision.category}   samples ${vision.scores.length}`,
  ];
  ctx.textAlign = "right";
  lines.forEach((l, i) => ctx.fillText(l, W - 20, 20 + th + 18 + i * 15));
  ctx.textAlign = "left";
}

// ── Input ──────────────────────────────────────────────────────────────────────

window.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "1" || k === "2" || k === "3") mode = +k;
  else if (CAT_KEYS[k]) vision.override = CAT_KEYS[k];
  else if (k === "a") vision.override = null;
  else if (k === "h") showHud = !showHud;
  else if (k === "d") debug = !debug;
});
window.addEventListener("dblclick", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
});
window.addEventListener("resize", resize);

// ── Boot ───────────────────────────────────────────────────────────────────────

const paperTile = makePaperTile();
const speckleTile = makeSpeckleTile();
await loadFonts();
resize();
requestAnimationFrame(loop);

const catParam = CAT_KEYS[(params.get("cat") || "").toLowerCase()] || null;
vision.init({
  demo: params.has("demo"),
  idle: params.has("idle"),
  still: params.has("still"),
  demoCat: catParam,
}).then(() => {
  if (catParam && !params.has("demo")) vision.override = catParam;
  window.__ready = true;
}).catch(e => {
  console.error(e);
  error = e && e.name === "NotAllowedError"
    ? "camera blocked — allow camera access, or open with ?demo"
    : String(e && e.message || e);
  window.__ready = true;
});
