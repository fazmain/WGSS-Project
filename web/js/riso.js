// riso.js — the printed-matter look: paper, inks, misregistration, halftone,
// and variable-width typography.
//
// Everything is drawn onto an off-white "paper" with globalCompositeOperation
// = "multiply", so overlapping inks mix the way risograph inks do
// (pink over yellow → red-orange, blue over yellow → green, pink over blue →
// purple).

export const PAPER = "#F3EEE3";

export const INK = {
  pink:   "#FF48B0",   // riso Fluorescent Pink
  blue:   "#0078BF",   // riso Blue
  yellow: "#FFE800",   // riso Yellow
  black:  "#26262E",
};

// Which ink carries the projection for each category. "mixed" overprints both.
export const CATEGORY_INKS = {
  feminine:  ["pink"],
  masculine: ["blue"],
  mixed:     ["pink", "blue"],
};

// ── Small math helpers ─────────────────────────────────────────────────────────
export const TAU = Math.PI * 2;
export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, k) => a + (b - a) * k;
export const rand = (a, b) => a + Math.random() * (b - a);
export const pick = arr => arr[Math.floor(Math.random() * arr.length)];
export const smoothstep = (a, b, v) => { const k = clamp((v - a) / (b - a), 0, 1); return k * k * (3 - 2 * k); };
export const easeOut = k => 1 - Math.pow(1 - clamp(k, 0, 1), 3);

// Deterministic PRNG so rows / letters keep a stable look between frames.
export function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Variable-width type ────────────────────────────────────────────────────────
// Canvas only accepts font-stretch keywords, so we register the variable font
// many times, each face pinned to one width via its `stretch` descriptor (the
// browser clamps the wdth axis to it). Weight stays continuous. The residual
// between steps is made up with a tiny horizontal scale (≤ 2.5%).
const W_MIN = 25, W_MAX = 150, W_STEP = 5;

export async function loadFonts() {
  const buf = await (await fetch("fonts/RobotoFlex.woff2")).arrayBuffer();
  const faces = [];
  for (let w = W_MIN; w <= W_MAX; w += W_STEP) {
    const f = new FontFace(`RF${w}`, buf, { weight: "100 1000", stretch: `${w}%` });
    document.fonts.add(f);
    faces.push(f.load());
  }
  await Promise.all(faces);
}

/** Set ctx.font for (size px, weight 100–1000, width 25–150). Returns x-scale. */
export function setFont(ctx, size, weight = 400, width = 100) {
  const w = clamp(width, W_MIN, W_MAX);
  const step = clamp(Math.round((w - W_MIN) / W_STEP) * W_STEP + W_MIN, W_MIN, W_MAX);
  ctx.font = `${Math.round(clamp(weight, 100, 1000))} ${size.toFixed(1)}px RF${step}`;
  return w / step;
}

export function measure(ctx, text, size, weight, width) {
  const sx = setFont(ctx, size, weight, width);
  return ctx.measureText(text).width * sx;
}

/** fillText with variable width. align: "left" | "center" | "right". */
export function drawText(ctx, text, x, y, size, weight, width, align = "center") {
  const sx = setFont(ctx, size, weight, width);
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(sx, 1);
  ctx.textAlign = align;
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

// ── Misregistration ────────────────────────────────────────────────────────────
// Each ink drum sits slightly off, and wanders. Motion shakes the press.
const REG = {
  pink:   { base: [-2.0,  1.2], seed: 1.3 },
  blue:   { base: [ 2.2, -1.0], seed: 4.1 },
  yellow: { base: [ 0.6,  2.4], seed: 7.7 },
  black:  { base: [ 0.0,  0.0], seed: 2.9 },
};

export function inkOffset(ink, t, motion = 0) {
  const r = REG[ink];
  const amp = 1.4 + motion * 7;
  return [
    r.base[0] + Math.sin(t * 0.37 + r.seed) * amp + Math.sin(t * 2.3 + r.seed * 3) * amp * 0.25,
    r.base[1] + Math.cos(t * 0.29 + r.seed * 2) * amp + Math.sin(t * 1.9 + r.seed) * amp * 0.25,
  ];
}

// ── Paper + ink texture ────────────────────────────────────────────────────────
function makeCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

/** Paper fibre/grain tile, composited with "multiply". */
export function makePaperTile(size = 512) {
  const c = makeCanvas(size, size);
  const g = c.getContext("2d");
  const img = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const n = 255 - Math.pow(Math.random(), 5) * 34 - Math.random() * 6;
    img.data[i * 4] = n; img.data[i * 4 + 1] = n; img.data[i * 4 + 2] = n - 2;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  // fibres
  g.lineCap = "round";
  for (let i = 0; i < 260; i++) {
    const x = Math.random() * size, y = Math.random() * size;
    const a = Math.random() * TAU, l = rand(6, 26);
    g.strokeStyle = `rgba(90,80,60,${rand(0.03, 0.09)})`;
    g.lineWidth = rand(0.4, 1.1);
    g.beginPath();
    g.moveTo(x, y);
    g.quadraticCurveTo(x + Math.cos(a + 0.6) * l * 0.5, y + Math.sin(a + 0.6) * l * 0.5,
                       x + Math.cos(a) * l, y + Math.sin(a) * l);
    g.stroke();
  }
  return c;
}

/** Ink voids: tiny paper-coloured specks where the drum didn't deposit ink. */
export function makeSpeckleTile(size = 512) {
  const c = makeCanvas(size, size);
  const g = c.getContext("2d");
  for (let i = 0; i < 2600; i++) {
    g.fillStyle = `rgba(250,247,240,${rand(0.35, 0.9)})`;
    g.beginPath();
    g.arc(Math.random() * size, Math.random() * size, Math.pow(Math.random(), 2) * 1.5 + 0.3, 0, TAU);
    g.fill();
  }
  return c;
}

// ── Halftone ───────────────────────────────────────────────────────────────────
/**
 * Fill a rotated halftone screen. `tone(x, y)` returns 0–1 ink coverage at a
 * screen point. All dots go into one path → one fill call.
 */
export function halftone(ctx, W, H, tone, { spacing = 9, angle = 15, gamma = 0.75, bounds = null } = {}) {
  const a = angle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const maxR = spacing * 0.66;
  const cx = W / 2, cy = H / 2;
  const n = Math.ceil(Math.hypot(W, H) / 2 / spacing) + 1;
  const [bx0, by0, bx1, by1] = bounds || [-maxR, -maxR, W + maxR, H + maxR];
  ctx.beginPath();
  for (let j = -n; j <= n; j++) {
    const gy = j * spacing;
    for (let i = -n; i <= n; i++) {
      const gx = i * spacing;
      const x = cx + gx * ca - gy * sa;
      const y = cy + gx * sa + gy * ca;
      if (x < bx0 || y < by0 || x > bx1 || y > by1) continue;
      const v = tone(x, y);
      if (v < 0.03) continue;
      const r = maxR * Math.pow(v, gamma);
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, TAU);
    }
  }
  ctx.fill();
}
