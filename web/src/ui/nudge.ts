/**
 * Frame stepping: the rules behind nudging a position off the detected frame.
 *
 * Detection picks one frame per position, and it is deterministic, but "the
 * frame at the top of the backswing" is a judgement the user can make better by
 * eye than a wrist-height minimum can. Stepping lets them correct it.
 *
 * Two things follow from a nudge, and both are the point of it:
 *
 *   - It scores. A similarity score is meant to measure technique, and a low
 *     one mixes real technique differences with the two swings being caught at
 *     slightly different moments. Nudging is the tool for removing the timing
 *     half, and it cannot do that job if scoring ignores where the user landed.
 *     The originally detected frame's score stays on screen beside it, because
 *     that one is reproducible and a nudged score is not.
 *   - It persists. A nudge is a correction, not a peek: leaving the position
 *     and coming back has to show the corrected frame, or lining two swings up
 *     carefully is wasted the moment you look away.
 *
 * The decoded window a nudge is chosen from is expensive and does not persist;
 * the chosen frame is one canvas and does. See NUDGE_HALF for that arithmetic.
 */

import { EVENTS, type EventName, type Pose } from "../core/constants";

export type Side = "yours" | "pro";

/** One decoded frame around a position, posed and ready to show. */
export interface NudgeFrame {
  canvas: HTMLCanvasElement;
  pose: Pose | null;
  timeSec: number;
}

/** A frame the user stepped to, with how far it sits from the detected one. */
export interface Nudge extends NudgeFrame {
  offset: number;
}

/**
 * The decoded frames around one position on one clip.
 *
 * `center` indexes the frame nearest the detected time, so an offset of 0 is
 * always the detected frame however the window was sliced or widened. `half` is
 * the half-width it was decoded at, which is what widening doubles.
 */
export interface NudgeWindow {
  frames: NudgeFrame[];
  center: number;
  half: number;
}

export type NudgeMap = Partial<Record<EventName, Nudge>>;
export type NudgeStore = Record<Side, NudgeMap>;

export const emptyNudges = (): NudgeStore => ({ yours: {}, pro: {} });

/**
 * Half-width of the decoded window, and the ceiling it will grow to.
 *
 * Memory sets both. A frame capped at 720 on its long side is ~1.2MB, and the
 * peak is one position across two clips: +/-16 is 33 frames a side, ~77MB.
 * Doubling to +/-32 is ~150MB, which is where keeping every frame of the clip
 * was ruled out in the first place, so growth stops there.
 *
 * It starts at 16 rather than the ceiling because most nudges are a frame or
 * two and never need the rest, and the whole clip is decoded to build a window.
 */
export const NUDGE_HALF = 16;
export const NUDGE_HALF_MAX = 32;

/** Long side for stepped frames; matches the tiles so crops line up. */
export const NUDGE_SIZE = 720;

/** Offsets the window can currently reach, relative to the detected frame. */
export function offsetRange(win: NudgeWindow): { lo: number; hi: number } {
  return { lo: -win.center, hi: win.frames.length - 1 - win.center };
}

export function clampOffset(win: NudgeWindow, offset: number): number {
  const { lo, hi } = offsetRange(win);
  return Math.max(lo, Math.min(offset, hi));
}

export function frameAt(win: NudgeWindow, offset: number): NudgeFrame | null {
  return win.frames[win.center + clampOffset(win, offset)] ?? null;
}

/** The half-width a widened window should be decoded at. */
export function nextHalf(win: NudgeWindow): number {
  return Math.min(win.half * 2, NUDGE_HALF_MAX);
}

/**
 * The half-width to decode the first window for a position at.
 *
 * Usually the default, but a position revisited after its window was dropped
 * starts at whatever offset it was left on, and that can be further out than
 * the default width if the window had been widened. Decoding narrower than the
 * stored offset would clamp the next step back towards the middle, quietly
 * undoing part of the user's correction.
 */
export function halfFor(offset: number): number {
  return Math.min(Math.max(NUDGE_HALF, Math.abs(offset)), NUDGE_HALF_MAX);
}

/**
 * Whether stepping past the window's edge should decode a wider one.
 *
 * Only when there is more clip to reach: a window truncated because the clip
 * ran out holds fewer frames on that side than it asked for, and re-decoding
 * would return exactly the same frames at the cost of another full decode.
 */
export function shouldWiden(
  win: NudgeWindow,
  offset: number,
  delta: number,
): boolean {
  if (win.half >= NUDGE_HALF_MAX) return false;
  const { lo, hi } = offsetRange(win);
  const target = offset + delta;
  if (target >= lo && target <= hi) return false;
  const room = delta < 0 ? win.center : win.frames.length - 1 - win.center;
  return room >= win.half;
}

/** Record (or, back at the detected frame, clear) the nudge for a position. */
export function setNudge(
  map: NudgeMap,
  name: EventName,
  frame: NudgeFrame | null,
  offset: number,
): NudgeMap {
  const next = { ...map };
  if (offset === 0 || !frame) delete next[name];
  else next[name] = { ...frame, offset };
  return next;
}

/** How far a position currently sits from its detected frame. */
export function offsetOf(map: NudgeMap, name: EventName): number {
  return map[name]?.offset ?? 0;
}

/**
 * The pose in play at each position: the nudged frame's where the user stepped,
 * otherwise the detected one. This is what gets scored.
 *
 * A nudged frame with no pose scores as null rather than falling back to the
 * detected pose. Scoring a frame that is not the one on screen is the exact
 * problem this feature exists to fix, so no score is the honest answer.
 */
export function posesInPlay(
  detected: Record<EventName, Pose | null>,
  nudges: NudgeMap,
): Record<EventName, Pose | null> {
  return Object.fromEntries(
    EVENTS.map((name) => {
      const nudge = nudges[name];
      return [name, nudge ? nudge.pose : detected[name]];
    }),
  ) as Record<EventName, Pose | null>;
}

/** Whether anything has been nudged, which is what puts the baseline on screen. */
export function anyNudged(store: NudgeStore): boolean {
  return Object.keys(store.yours).length > 0 || Object.keys(store.pro).length > 0;
}
