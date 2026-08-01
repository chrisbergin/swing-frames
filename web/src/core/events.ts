/** Wrist tracking and swing-position detection.
 *
 * Ported from smooth(), wrist_track(), crossing(), detect_events() and the
 * frame-selection half of refine_impact() in swing_frames.py.
 *
 * Detection is purely position-based: hand speed is never used, because it is
 * zero on freeze frames and spikes at cuts in pause-and-step analysis videos,
 * which broke two earlier speed-based versions of this.
 */

import {
  EVENTS,
  LM,
  type EventFrames,
  type EventName,
  type Pose,
} from "./constants";

/** Thrown when a clip cannot be analysed. The message is shown to the user. */
export class SwingDetectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SwingDetectionError";
  }
}

/**
 * Smoothing width tied to frame rate, so roughly the same slice of real time
 * is averaged whether the clip is 30fps or 240fps slo-mo.
 */
export function smoothWindow(fps: number): number {
  return Math.max(3, Math.trunc(fps / 15));
}

/**
 * Moving average with the signal's end values extended past the edges, so the
 * output keeps full length instead of sagging toward zero at the ends.
 */
export function smooth(x: readonly number[], window: number): number[] {
  const n = x.length;
  if (n === 0) return [];
  const w = Math.max(3, window | 1); // odd, at least 3
  const half = Math.floor(w / 2);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let k = 0; k < w; k++) {
      // Index into the conceptually edge-padded signal.
      const j = Math.min(n - 1, Math.max(0, i + k - half));
      sum += x[j];
    }
    out[i] = sum / w;
  }
  return out;
}

/** Linear-interpolated percentile, matching numpy's default method. */
export function percentile(values: readonly number[], p: number): number {
  const s = [...values].sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

/** Fill NaN gaps by linear interpolation, clamping beyond the outermost samples. */
function interpolateGaps(v: readonly number[]): number[] {
  const n = v.length;
  const valid: number[] = [];
  for (let i = 0; i < n; i++) if (!Number.isNaN(v[i])) valid.push(i);
  const first = valid[0];
  const last = valid[valid.length - 1];

  const out = new Array<number>(n);
  let ptr = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isNaN(v[i])) {
      out[i] = v[i];
      continue;
    }
    if (i < first) {
      out[i] = v[first];
      continue;
    }
    if (i > last) {
      out[i] = v[last];
      continue;
    }
    while (ptr < valid.length - 1 && valid[ptr + 1] <= i) ptr++;
    const a = valid[ptr];
    const b = valid[ptr + 1];
    out[i] = v[a] + ((v[b] - v[a]) * (i - a)) / (b - a);
  }
  return out;
}

/** Index of the smallest value in [start, end), first occurrence on ties. */
function argMin(ys: readonly number[], start: number, end: number): number {
  let best = start;
  for (let i = start + 1; i < end; i++) if (ys[i] < ys[best]) best = i;
  return best;
}

/** Index of the largest value in [start, end), first occurrence on ties. */
function argMax(ys: readonly number[], start: number, end: number): number {
  let best = start;
  for (let i = start + 1; i < end; i++) if (ys[i] > ys[best]) best = i;
  return best;
}

/**
 * Wrist-midpoint trajectory with gaps interpolated and smoothing applied.
 *
 * Only `ys` drives detection: the whole swing structure is legible in hand
 * height alone. `xs` is tracked alongside it for overlays and future use.
 */
export function wristTrack(
  poses: readonly (Pose | null)[],
  fps: number,
): { xs: number[]; ys: number[] } {
  const n = poses.length;
  const xs = new Array<number>(n).fill(NaN);
  const ys = new Array<number>(n).fill(NaN);

  let validCount = 0;
  for (let i = 0; i < n; i++) {
    const p = poses[i];
    if (!p) continue;
    xs[i] = (p[LM.L_WRIST].x + p[LM.R_WRIST].x) / 2;
    ys[i] = (p[LM.L_WRIST].y + p[LM.R_WRIST].y) / 2;
    validCount++;
  }

  if (validCount < Math.max(10, n * 0.5)) {
    throw new SwingDetectionError(
      `Found a golfer in only ${validCount} of ${n} sampled frames. Usual ` +
        `causes, most likely first: the golfer is partly out of frame (head ` +
        `or feet cropped), the clip is very low resolution (screen recordings ` +
        `especially; prefer a full-resolution download), or the video decodes ` +
        `sideways (set Rotation). The Heavy model sometimes helps borderline ` +
        `clips.`,
    );
  }

  const win = smoothWindow(fps);
  return {
    xs: smooth(interpolateGaps(xs), win),
    ys: smooth(interpolateGaps(ys), win),
  };
}

/**
 * First frame in [start, end) where `ys` crosses `level`.
 *
 * "up" means hands rising, which is y decreasing, because y grows downward.
 */
export function crossing(
  ys: readonly number[],
  start: number,
  end: number,
  level: number,
  direction: "up" | "down",
): number | null {
  for (let i = Math.max(0, start); i < Math.min(end, ys.length); i++) {
    if (direction === "up" && ys[i] <= level) return i;
    if (direction === "down" && ys[i] >= level) return i;
  }
  return null;
}

/**
 * Map the 8 swing positions to frame indices from wrist height alone.
 *
 * Structure: the hands go high twice, at the top of the backswing and again
 * on the finish hold. Find those two episodes, then anchor everything else
 * around and between them.
 */
export function detectEvents(ys: readonly number[], fps: number): EventFrames {
  const n = ys.length;
  const baseline = percentile(ys, 97); // hands-down (address) height
  let lowest = Infinity;
  for (const y of ys) if (y < lowest) lowest = y;
  const rng = baseline - lowest; // positive: high hands = small y

  if (rng < 20) {
    throw new SwingDetectionError(
      "No backswing motion detected. Is the clip trimmed to one swing?",
    );
  }

  // Contiguous runs of "hands high", ignoring blips shorter than 3 frames.
  const highLevel = baseline - 0.6 * rng;
  const episodes: Array<[number, number]> = [];
  let runStart: number | null = null;
  for (let i = 0; i < n; i++) {
    const high = ys[i] < highLevel;
    if (high && runStart === null) runStart = i;
    else if (!high && runStart !== null) {
      episodes.push([runStart, i]);
      runStart = null;
    }
  }
  if (runStart !== null) episodes.push([runStart, n]);
  const kept = episodes.filter(([a, b]) => b - a >= 3);

  if (kept.length === 0) {
    throw new SwingDetectionError(
      "No backswing motion detected. Is the clip trimmed to one swing?",
    );
  }

  const [e1a, e1b] = kept[0];
  const top = argMin(ys, e1a, e1b);

  let finish: number;
  let impactZoneEnd: number;
  if (kept.length > 1) {
    const [e2a, e2b] = kept[kept.length - 1];
    finish = argMin(ys, e2a, e2b);
    impactZoneEnd = e2a;
  } else {
    // Clip ends before a distinct finish hold (e.g. cut right after impact).
    finish = n - 1;
    impactZoneEnd = n;
  }

  // Address: if the clip starts with the hands already set (the recommended
  // trimming), frame 0 is the cleanest address. Otherwise take the last
  // near-baseline frame, backed off by the smoothing window so the takeaway
  // boundary, smeared by smoothing at hard cuts, is not the one picked.
  const nearBaseline = baseline - 0.05 * rng;
  const addr: number[] = [];
  for (let i = 0; i < top; i++) if (ys[i] >= nearBaseline) addr.push(i);
  const win = smoothWindow(fps);
  const address =
    addr.length === 0 || addr[0] === 0
      ? 0
      : Math.max(addr[addr.length - 1] - win, addr[0]);

  // Impact: hands at their lowest between the top and the finish episode. At
  // regular frame rates this lands 1-2 frames after the strike, because VIDEO
  // mode's temporal prior lags the motion-blurred hands through the hitting
  // zone; selectImpactFrame() re-measures the window afterwards.
  const impact = argMax(ys, top, impactZoneEnd);

  // Downswing and follow-through mids are placed by time (the frame halfway
  // through the phase), not by a wrist-height crossing. A height crossing fires
  // at a different point on two different swings, so the club lands on opposite
  // sides in a comparison; the halfway-by-time frame is tempo-normalized and
  // corresponds across swings. Those phases are also fast and noisy, where a
  // crossing is least reliable.
  const found: Record<EventName, number | null> = {
    address,
    toe_up: crossing(ys, address, top, baseline - 0.3 * rng, "up"),
    mid_backswing: crossing(ys, address, top, baseline - 0.7 * rng, "up"),
    top,
    mid_downswing: Math.round(top + 0.5 * (impact - top)),
    impact,
    mid_follow_through: Math.round(impact + 0.5 * (finish - impact)),
    finish,
  };

  // Any crossing that was never reached falls back to the midpoint of its
  // resolved neighbours, so all 8 positions always come back populated.
  const order = EVENTS.map((e) => found[e]);
  for (let i = 0; i < order.length; i++) {
    if (order[i] !== null) continue;
    let prev = 0;
    for (let j = i - 1; j >= 0; j--) {
      if (order[j] !== null) {
        prev = order[j]!;
        break;
      }
    }
    let next = n - 1;
    for (let j = i + 1; j < order.length; j++) {
      if (order[j] !== null) {
        next = order[j]!;
        break;
      }
    }
    order[i] = Math.floor((prev + next) / 2);
  }

  return Object.fromEntries(
    EVENTS.map((e, i) => [e, order[i] as number]),
  ) as EventFrames;
}

/**
 * The narrow window worth re-measuring for impact, searching mostly backwards
 * because the tracked impact is only ever late, never early.
 */
export function impactRefineWindow(
  events: EventFrames,
  fps: number,
  nFrames: number,
): { lo: number; hi: number } {
  const win = smoothWindow(fps);
  return {
    lo: Math.max(events.top, events.impact - 2 * win),
    hi: Math.min(nFrames, events.impact + win + 1),
  };
}

/**
 * Pick the true impact frame from independently re-measured wrist heights.
 *
 * Ball contact comes just after the hands' lowest point (a teed ball is struck
 * on the upswing), and consecutive frames near the bottom measure within noise
 * of each other, so take the last frame within a hair of the lowest.
 */
export function selectImpactFrame(
  measured: ReadonlyArray<{ frame: number; y: number }>,
  fallback: number,
): number {
  if (measured.length === 0) return fallback;
  let bottom = -Infinity;
  let peak = Infinity;
  for (const m of measured) {
    if (m.y > bottom) bottom = m.y;
    if (m.y < peak) peak = m.y;
  }
  const eps = 0.05 * (bottom - peak);
  let best = fallback;
  let seen = false;
  for (const m of measured) {
    if (m.y >= bottom - eps && (!seen || m.frame > best)) {
      best = m.frame;
      seen = true;
    }
  }
  return best;
}
