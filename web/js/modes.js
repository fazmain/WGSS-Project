// modes.js — three kinetic-typography treatments of the same reading.
//
//   1 FILL     the body is made of words: rows of labels slide through the
//              silhouette, clipped to it. Category decides the typeface's
//              width/weight, so a feminine reading literally takes up less
//              space and a masculine one crowds the body.
//   2 PERFORM  labels float around the body and *act out* the expectation
//              over their lifetime — squeezing thin and light, or swelling
//              wide and heavy. Unresolved readings can't hold a single form.
//   3 HOLD     stand still and letters fly in and stick to you, word by
//              word. Move and they shake loose and fall away.
//
// The body itself is always a yellow halftone (drawn in main.js); the
// projected labels are in the category's ink.

import { POOLS } from "./pools.js";
import {
  INK, CATEGORY_INKS, TAU, clamp, lerp, rand, pick, seeded, smoothstep, easeOut,
  setFont, measure, drawText, inkOffset,
} from "./riso.js";
import { LM } from "./vision.js";

// ── Typographic stereotypes ────────────────────────────────────────────────────
// How each reading shapes the type. Width is the font's wdth axis (25–150).
const FEM  = { weight: 160, width: 28 };
const MASC = { weight: 960, width: 145 };

/** Per-letter form for an unresolved reading — never settles. */
function mixedLetter(i, t, seed) {
  const ph = t * 1.5 + i * 0.55 + seed;
  return {
    width: 87 + 62 * Math.sin(ph),
    weight: 560 + 420 * Math.sin(ph * 0.8 + 1.3),
    ink: (i + Math.floor(t * 0.8 + seed)) % 2 ? "pink" : "blue",
  };
}

function inkFor(cat) { return cat === "masculine" ? "blue" : "pink"; }

// ── Anchors ────────────────────────────────────────────────────────────────────

/** Named screen-space points on the body that labels orbit / stick to. */
export function anchorsFrom(lm, toScreen, H) {
  const A = {};
  if (!lm) return A;
  const P = i => (lm[i] && lm[i].v > 0.35 ? toScreen(lm[i].x, lm[i].y) : null);
  const nose = P(LM.nose), ls = P(LM.lSh), rs = P(LM.rSh);
  const lh = P(LM.lHip), rh = P(LM.rHip), lw = P(LM.lWr), rw = P(LM.rWr);
  const sw = ls && rs ? Math.hypot(rs[0] - ls[0], rs[1] - ls[1]) : H * 0.2;
  A.sw = sw;
  if (nose) A.head = [nose[0], nose[1] - sw * 0.25];
  if (ls) A.ls = ls;
  if (rs) A.rs = rs;
  if (lw) A.lw = lw;
  if (rw) A.rw = rw;
  const pts = [ls, rs, lh, rh].filter(Boolean);
  if (pts.length) A.torso = [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length];
  const hips = [lh, rh].filter(Boolean);
  if (hips.length) A.hips = [hips.reduce((a, p) => a + p[0], 0) / hips.length, hips.reduce((a, p) => a + p[1], 0) / hips.length];
  if (ls && rs) {
    const mx = (ls[0] + rs[0]) / 2, my = (ls[1] + rs[1]) / 2, hw = Math.abs(rs[0] - ls[0]) * 0.9;
    A.wideL = [mx - hw * 1.2, my + sw * 0.2];
    A.wideR = [mx + hw * 1.2, my + sw * 0.2];
    A.upL = [mx - hw * 0.7, my - sw * 0.45];
    A.upR = [mx + hw * 0.7, my - sw * 0.45];
  }
  return A;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1 · FILL
// ═══════════════════════════════════════════════════════════════════════════════

export class Fill {
  constructor() {
    this.cat = null;
    this.fade = 0;
    this.leaving = false;
    this.rows = new Map();     // key → { text, dir, speed, a, b }
    this.offsets = new Map();  // key → scroll offset px
    this.size = 0;
  }

  _row(ink, r) {
    const key = `${ink}:${r}`;
    let row = this.rows.get(key);
    if (!row) {
      const rnd = seeded(r * 7919 + (ink === "blue" ? 104729 : 1) + this.cat.length * 31);
      const pool = POOLS[this.cat].map(s => s.toUpperCase());
      for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
      row = {
        words: pool.map(w => w + "  ·  "),
        dir: r % 2 ? 1 : -1,
        speed: 0.6 + rnd() * 0.9,
        a: rnd(), b: rnd(),
        widths: new Map(),   // style key → [word widths, total]
      };
      this.rows.set(key, row);
    }
    return row;
  }

  /** Word advances for a row at a (quantised) style — cached. */
  _widths(ctx, row, size, st) {
    const key = `${Math.round(size)}|${st.weight}|${st.width}`;
    let w = row.widths.get(key);
    if (!w) {
      const ws = row.words.map(s => measure(ctx, s, size, st.weight, st.width));
      w = [ws, ws.reduce((a, b) => a + b, 0)];
      if (row.widths.size > 64) row.widths.clear();
      row.widths.set(key, w);
    }
    return w;
  }

  _style(ink, row, r, t) {
    const fem = { weight: FEM.weight + row.a * 140, width: FEM.width + row.b * 18 };
    const masc = { weight: MASC.weight - row.a * 90, width: MASC.width - row.b * 20 };
    if (this.cat === "feminine") return fem;
    if (this.cat === "masculine") return masc;
    // unresolved: each row keeps trying on the other form
    // (quantised so word widths can be cached — 16 steps reads as smooth)
    const k = Math.round((0.5 + 0.5 * Math.sin(t * 0.7 + r * 0.9 + (ink === "blue" ? Math.PI : 0))) * 16) / 16;
    const from = ink === "pink" ? fem : masc, to = ink === "pink" ? masc : fem;
    return { weight: lerp(from.weight, to.weight, k * 0.85), width: lerp(from.width, to.width, k * 0.85) };
  }

  frame(F) {
    const { ctx, lctx, layer, W, H, u, t, dt } = F;
    if (F.cat !== this.cat && !this.leaving) this.leaving = this.cat !== null;
    if (this.cat === null) this.cat = F.cat;
    if (this.leaving) {
      this.fade -= dt * 3;
      if (this.fade <= 0) { this.fade = 0; this.leaving = false; this.cat = F.cat; this.rows.clear(); }
    } else {
      this.fade = Math.min(1, this.fade + dt * 1.5);
    }
    if (!F.bounds || this.fade <= 0) return;

    const target = lerp(30, 66, F.closeness) * u;
    this.size = this.size ? lerp(this.size, target, 1 - Math.exp(-dt * 1.2)) : target;
    const size = Math.round(this.size);
    const inks = CATEGORY_INKS[this.cat];
    const lead = size * (inks.length > 1 ? 1.0 : 0.9);
    const [bx0, by0, bx1, by1] = F.bounds;

    for (const ink of inks) {
      lctx.save();
      lctx.setTransform(1, 0, 0, 1, 0, 0);
      lctx.clearRect(0, 0, layer.width, layer.height);
      lctx.restore();
      lctx.fillStyle = INK[ink];
      lctx.textBaseline = "alphabetic";
      const shift = ink === "blue" && inks.length > 1 ? lead * 0.5 : 0;

      const rows = Math.ceil(H / lead) + 2;
      for (let r = 0; r < rows; r++) {
        const y = r * lead + size * 0.8 + shift;
        if (y < by0 - size || y - size > by1) continue;   // row misses the body
        const row = this._row(ink, r);
        const st = this._style(ink, row, r, t);
        const key = `${ink}:${r}`;
        const speed = (16 + 260 * F.motion) * u * row.speed * row.dir;
        const off = (this.offsets.get(key) || rand(0, 2000)) + speed * dt;
        this.offsets.set(key, off);

        // Only set the words that overlap the body — the rest is clipped anyway
        const [ws, rep] = this._widths(lctx, row, size, st);
        if (rep < 1) continue;
        let x = (((off % rep) + rep) % rep) - rep;
        for (let i = 0; x < bx1; i = (i + 1) % ws.length) {
          if (x + ws[i] > bx0) drawText(lctx, row.words[i], x, y, size, st.weight, st.width, "left");
          x += ws[i];
        }
      }

      // Clip to the silhouette
      lctx.globalCompositeOperation = "destination-in";
      lctx.drawImage(F.maskCanvas, ...F.maskRect);
      lctx.globalCompositeOperation = "source-over";

      const o = inkOffset(ink, t, F.motion);
      ctx.globalCompositeOperation = "multiply";
      ctx.globalAlpha = this.fade;
      ctx.drawImage(layer, o[0], o[1], W, H);
      ctx.globalAlpha = 1;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2 · PERFORM
// ═══════════════════════════════════════════════════════════════════════════════

const ORBIT_SLOTS = ["head", "head", "ls", "rs", "torso", "torso", "hips",
                     "wideL", "wideR", "upL", "upR", "lw", "rw"];

class Label {
  constructor(text, cat, u) {
    this.text = text.toUpperCase();
    this.cat = cat;
    this.key = pick(ORBIT_SLOTS);
    this.size = rand(34, 74) * u;
    const a = rand(0, TAU), d = rand(60, 160) * u;
    this.base = [Math.cos(a) * d, Math.sin(a) * d];
    this.df = [rand(0.1, 0.3), rand(0.1, 0.3)];
    this.dp = [rand(0, TAU), rand(0, TAU)];
    this.da = [rand(14, 40) * u, rand(10, 30) * u];
    this.vel = [0, 0];
    this.pos = null;
    this.age = 0;
    this.life = rand(7, 13);
    this.alpha = 0;
    this.out = false;
    this.seed = rand(0, 100);
    this.hw = 40; this.hh = this.size * 0.45;
  }

  /** Over its life the word acts out the expectation. */
  form() {
    const k = easeOut(this.age / 5.5);
    if (this.cat === "feminine")
      return { weight: lerp(430, 110, k), width: lerp(100, 25, k), scale: lerp(1, 0.7, k), reach: lerp(1, 0.45, k) };
    if (this.cat === "masculine")
      return { weight: lerp(430, 1000, k), width: lerp(100, 150, k), scale: lerp(1, 1.4, k), reach: lerp(1, 1.25, k) };
    return { scale: 1, reach: 1 + 0.25 * Math.sin(this.age * 1.3 + this.seed) };
  }

  update(F, A) {
    const dt = F.dt;
    this.age += dt;
    if (this.age > this.life || !F.present) this.out = true;
    this.alpha = this.out ? Math.max(0, this.alpha - dt * 1.4) : Math.min(1, this.alpha + dt * 1.6);

    if (F.spike > 0) {
      const a = rand(0, TAU), s = 520 * F.u * F.spike;
      this.vel[0] += Math.cos(a) * s; this.vel[1] += Math.sin(a) * s;
    }
    const decay = Math.exp(-4 * dt);
    this.vel[0] *= decay; this.vel[1] *= decay;

    const anchor = A[this.key] || A.torso;
    if (!anchor) return;
    const f = this.form();
    const tx = anchor[0] + this.base[0] * f.reach + this.da[0] * Math.sin(TAU * this.df[0] * this.age + this.dp[0]);
    const ty = anchor[1] + this.base[1] * f.reach + this.da[1] * Math.sin(TAU * this.df[1] * this.age + this.dp[1]);
    const target = [clamp(tx, this.hw + 6, F.W - this.hw - 6), clamp(ty, this.hh + 6, F.H - this.hh - 6)];
    if (!this.pos) this.pos = target.slice();
    this.pos[0] += this.vel[0] * dt; this.pos[1] += this.vel[1] * dt;
    const k = Math.min(3.2 * dt, 1);
    this.pos[0] += (target[0] - this.pos[0]) * k;
    this.pos[1] += (target[1] - this.pos[1]) * k;
  }

  draw(F) {
    if (!this.pos || this.alpha <= 0.01) return;
    const { ctx, t } = F;
    const f = this.form();
    let size = this.size * f.scale;
    ctx.globalAlpha = this.alpha;
    if (this.cat !== "mixed") {
      const ink = inkFor(this.cat);
      const o = inkOffset(ink, t, F.motion);
      // Swelling words may crowd the frame, but never outgrow it
      let w = measure(ctx, this.text, size, f.weight, f.width);
      const maxW = F.W * 0.6;
      if (w > maxW) { size *= maxW / w; w = maxW; }
      ctx.fillStyle = INK[ink];
      drawText(ctx, this.text, this.pos[0] + o[0], this.pos[1] + o[1], size, f.weight, f.width, "center");
      this.hw = w / 2 + 8;
    } else {
      this.hw = drawLetters(ctx, this.text, this.pos[0], this.pos[1], size, i => mixedLetter(i, this.age, this.seed), t, F.motion) / 2 + 8;
    }
    this.hh = size * 0.42 + 6;
    ctx.globalAlpha = 1;
  }
}

/** Draw a word letter by letter with per-letter form + ink, centred on x. */
function drawLetters(ctx, text, x, y, size, formAt, t, motion) {
  const forms = [];
  let total = 0;
  for (let i = 0; i < text.length; i++) {
    const f = formAt(i);
    f.adv = measure(ctx, text[i], size, f.weight, f.width);
    forms.push(f);
    total += f.adv;
  }
  let cx = x - total / 2;
  for (let i = 0; i < text.length; i++) {
    const f = forms[i];
    if (text[i] !== " ") {
      const o = inkOffset(f.ink, t, motion);
      ctx.fillStyle = INK[f.ink];
      drawText(ctx, text[i], cx + o[0], y + o[1], size, f.weight, f.width, "left");
    }
    cx += f.adv;
  }
  return total;
}

/** Soft elliptical repulsion so words stay legible. */
function separate(labels, W, H, dt) {
  const live = labels.filter(l => l.pos && l.alpha > 0.05);
  const push = Math.min(8 * dt, 0.5);
  for (let i = 0; i < live.length; i++) {
    const a = live[i];
    for (let j = i + 1; j < live.length; j++) {
      const b = live[j];
      const dx = b.pos[0] - a.pos[0], dy = b.pos[1] - a.pos[1];
      const ex = a.hw + b.hw, ey = a.hh + b.hh;
      let nd = (dx / ex) ** 2 + (dy / ey) ** 2;
      if (nd >= 1 || nd < 1e-6) continue;
      nd = Math.sqrt(nd);
      const s = (1 - nd) * push;
      const sx = (dx / nd) * s * 0.5, sy = (dy / nd) * s * 0.5;
      a.pos[0] -= sx; a.pos[1] -= sy; b.pos[0] += sx; b.pos[1] += sy;
    }
    a.pos[0] = clamp(a.pos[0], a.hw * 0.6, W - a.hw * 0.6);
    a.pos[1] = clamp(a.pos[1], a.hh, H - a.hh);
  }
}

export class Perform {
  constructor() { this.labels = []; this.cat = null; this.cool = 0; }

  frame(F) {
    const { ctx, dt, u } = F;
    if (F.cat !== this.cat) {
      this.cat = F.cat;
      for (const l of this.labels) l.out = true;
    }
    // Feminine readings get more, smaller words; masculine fewer, bigger ones
    const share = this.cat === "feminine" ? 1.3 : this.cat === "masculine" ? 0.65 : 1;
    const target = F.present ? Math.round(lerp(8, 18, F.closeness) * share) : 0;
    const live = this.labels.filter(l => !l.out);
    this.cool -= dt;
    if (F.present && live.length < target && this.cool <= 0) {
      const onScreen = new Set(live.map(l => l.text));
      const pool = POOLS[this.cat].filter(s => !onScreen.has(s.toUpperCase()));
      this.labels.push(new Label(pick(pool.length ? pool : POOLS[this.cat]), this.cat, u));
      this.cool = 0.22;
    }
    if (live.length > target) live.slice(0, live.length - target).forEach(l => (l.out = true));

    for (const l of this.labels) l.update(F, F.anchors);
    separate(this.labels, F.W, F.H, dt);

    ctx.globalCompositeOperation = "multiply";
    for (const l of this.labels) l.draw(F);
    this.labels = this.labels.filter(l => !(l.out && l.alpha <= 0));
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3 · HOLD
// ═══════════════════════════════════════════════════════════════════════════════

const STICK_SLOTS = ["head", "torso", "torso", "torso", "hips", "ls", "rs", "upL", "upR"];

class StuckWord {
  /** `taken`: [{x, y, hw, hh}] resting spots of words already on the body. */
  constructor(text, cat, F, taken) {
    this.text = text.toUpperCase();
    this.cat = cat;
    this.alpha = 1;
    this.free = false;
    this.age = 0;
    const A = F.anchors, u = F.u;
    const size = lerp(24, 50, F.closeness) * u * rand(0.8, 1.3);

    // Lay out glyphs
    const seed = rand(0, 100);
    const forms = [];
    let total = 0;
    for (let i = 0; i < this.text.length; i++) {
      let f;
      // (a touch heavier than FEM — these letters are small)
      if (cat === "feminine") f = { weight: 300, width: 36, ink: "pink" };
      else if (cat === "masculine") f = { weight: MASC.weight, width: MASC.width, ink: "blue" };
      else { const m = mixedLetter(i, 0, seed); f = { weight: m.weight, width: m.width, ink: m.ink }; }
      f.adv = measure(F.ctx, this.text[i], size, f.weight, f.width);
      forms.push(f);
      total += f.adv;
    }
    this.hw = total / 2 + 4 * u;
    this.hh = size * 0.5;

    // Find a resting spot on the body, clear of the words already there
    const keys = STICK_SLOTS.filter(k => A[k]);
    const R = (A.sw || F.H * 0.2) * 0.8;
    let best = null, bestScore = -1;
    for (let i = 0; i < 30; i++) {
      const key = pick(keys);
      const o = [rand(-R, R), rand(-R * 0.9, R * 0.9)];
      const x = A[key][0] + o[0], y = A[key][1] + o[1];
      const onBody = F.mask(x, y);
      if (onBody < 0.5) continue;
      const clear = taken.every(s => ((x - s.x) / (this.hw + s.hw)) ** 2 + ((y - s.y) / (this.hh + s.hh)) ** 2 > 1);
      const score = onBody + (clear ? 1 : 0);
      if (score > bestScore) { best = { key, o }; bestScore = score; }
      if (clear) break;
    }
    this.key = best ? best.key : A.torso ? "torso" : keys[0];
    this.off = best ? best.o : [0, 0];
    const anchor = A[this.key] || [F.W / 2, F.H / 2];

    // Letters stream in from somewhere off the body
    const a = rand(0, TAU), far = Math.max(F.W, F.H) * 0.7;
    const sx = anchor[0] + Math.cos(a) * far, sy = anchor[1] + Math.sin(a) * far;
    let cx = -total / 2;
    this.glyphs = forms.map((f, i) => {
      const g = {
        ch: this.text[i], size, ...f,
        hx: cx + f.adv / 2, hy: 0,
        x: sx + rand(-60, 60) * u, y: sy + rand(-60, 60) * u,
        vx: 0, vy: 0, delay: i * 0.035,
      };
      cx += f.adv;
      return g;
    });
  }

  spot(A) {
    const a = A[this.key] || A.torso;
    return a ? { x: a[0] + this.off[0], y: a[1] + this.off[1], hw: this.hw, hh: this.hh } : null;
  }

  release(power, u) {
    if (this.free) return;
    this.free = true;
    for (const g of this.glyphs) {
      const a = rand(0, TAU), s = rand(250, 800) * u * power;
      g.vx += Math.cos(a) * s; g.vy += Math.sin(a) * s - 200 * u * power;
    }
  }

  update(F, cohesion) {
    const { dt, u } = F;
    this.age += dt;
    if (this.free) this.alpha = Math.max(0, this.alpha - dt / 1.8);
    const anchor = F.anchors[this.key] || F.anchors.torso;
    for (const g of this.glyphs) {
      if (this.age < g.delay) continue;
      if (!this.free && anchor) {
        const hx = anchor[0] + this.off[0] + g.hx;
        const hy = anchor[1] + this.off[1] + g.hy;
        const k = 38 * cohesion * Math.min(1, (this.age - g.delay) * 1.4);
        g.vx += (hx - g.x) * k * dt;
        g.vy += (hy - g.y) * k * dt;
        const damp = Math.exp(-6.5 * dt);
        g.vx *= damp; g.vy *= damp;
        // restlessness while the body moves
        const j = (1 - cohesion) * 900 * u * dt;
        g.vx += rand(-j, j); g.vy += rand(-j, j);
      } else {
        g.vy += 520 * u * dt;          // shed labels fall
        const damp = Math.exp(-0.8 * dt);
        g.vx *= damp; g.vy *= damp;
      }
      g.x += g.vx * dt; g.y += g.vy * dt;
    }
  }

  draw(F) {
    const { ctx, t } = F;
    ctx.globalAlpha = this.alpha;
    for (const g of this.glyphs) {
      if (this.age < g.delay || g.ch === " ") continue;
      const o = inkOffset(g.ink, t, F.motion);
      ctx.fillStyle = INK[g.ink];
      drawText(ctx, g.ch, g.x + o[0], g.y + o[1] + g.size * 0.35, g.size, g.weight, g.width, "center");
    }
    ctx.globalAlpha = 1;
  }
}

export class Hold {
  constructor() { this.words = []; this.cat = null; this.still = 0; this.cool = 0; this.churn = 0; }

  frame(F) {
    const { ctx, dt, u } = F;
    const M = F.motion;
    const attached = () => this.words.filter(w => !w.free);

    if (F.cat !== this.cat || !F.present) {
      this.cat = F.cat;
      for (const w of this.words) w.release(0.4, u);
      this.still = 0;
    }
    if (M > 0.4) {
      for (const w of attached()) w.release(0.5 + M, u);
      this.still = 0;
    } else if (M < 0.16 && F.present) {
      this.still += dt;
    }

    const maxN = Math.round(lerp(10, 24, F.closeness));
    const target = F.present && F.anchors.torso ? Math.min(maxN, Math.floor(1 + this.still * 1.6)) : 0;
    this.cool -= dt;
    const n = attached().length;
    if (n < target && this.cool <= 0) {
      const onBody = new Set(attached().map(w => w.text));
      const pool = POOLS[this.cat].filter(s => !onBody.has(s.toUpperCase()));
      const taken = attached().map(w => w.spot(F.anchors)).filter(Boolean);
      this.words.push(new StuckWord(pick(pool.length ? pool : POOLS[this.cat]), this.cat, F, taken));
      this.cool = 0.32;
    }
    // Keep the vocabulary churning once the body is covered
    this.churn += dt;
    if (n >= maxN && this.churn > 4) { attached()[0].release(0.3, u); this.churn = 0; }

    const cohesion = (1 - smoothstep(0.08, 0.42, M)) ** 2;
    ctx.globalCompositeOperation = "multiply";
    for (const w of this.words) { w.update(F, cohesion); w.draw(F); }
    this.words = this.words.filter(w => w.alpha > 0);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Idle — nobody in frame
// ═══════════════════════════════════════════════════════════════════════════════

export function drawIdle(F, alpha) {
  const { ctx, W, H, u, t } = F;
  ctx.globalCompositeOperation = "multiply";
  ctx.globalAlpha = alpha;
  const lines = ["PROJECTED", "EXPECTATIONS"];
  const base = measure(ctx, "EXPECTATIONS", 100, 820, 88);
  const size = Math.min((W * 0.86) / base * 100, H * 0.24);
  const lineH = size * 0.92;
  const y0 = H * 0.47 - lineH * 0.5;

  for (const [ink, phase] of [["pink", 0], ["blue", 0.45]]) {
    lines.forEach((line, li) => {
      drawLetters(ctx, line, W / 2, y0 + li * lineH, size, i => ({
        width: 88 + 46 * Math.sin(t * 0.75 - i * 0.42 - li * 0.8 + phase),
        weight: 780 + 200 * Math.sin(t * 0.5 - i * 0.3 + phase),
        ink,
      }), t, 0);
    });
  }

  ctx.fillStyle = INK.black;
  ctx.letterSpacing = `${(3 * u).toFixed(1)}px`;
  drawText(ctx, "STEP INTO THE FRAME", W / 2, y0 + lineH * 1.75, Math.max(13, 17 * u), 600, 120, "center");
  ctx.letterSpacing = `${(1 * u).toFixed(1)}px`;
  drawText(ctx, "the system is waiting to read you", W / 2, y0 + lineH * 1.75 + 28 * u, Math.max(12, 14 * u), 350, 100, "center");
  ctx.letterSpacing = "0px";
  ctx.globalAlpha = 1;
}
