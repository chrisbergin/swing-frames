import { describe, expect, it } from "vitest";
import { EVENTS, LM, type EventName, type Pose } from "../core/constants";
import {
  anyNudged,
  clampOffset,
  emptyNudges,
  frameAt,
  halfFor,
  nextHalf,
  offsetOf,
  offsetRange,
  posesInPlay,
  setNudge,
  shouldWiden,
  NUDGE_HALF,
  NUDGE_HALF_MAX,
  type NudgeFrame,
  type NudgeWindow,
} from "./nudge";

/** Frames only need identity and a time here; nothing touches the canvas. */
function frame(timeSec: number, pose: Pose | null = null): NudgeFrame {
  return { canvas: {} as HTMLCanvasElement, pose, timeSec };
}

/**
 * A window of `half` frames either side of the centre, optionally truncated as
 * it would be when the position sits near the start or end of the clip.
 */
function window({
  half = NUDGE_HALF,
  before = half,
  after = half,
}: { half?: number; before?: number; after?: number } = {}): NudgeWindow {
  const frames: NudgeFrame[] = [];
  for (let i = -before; i <= after; i++) frames.push(frame(1 + i * 0.03));
  return { frames, center: before, half };
}

function pose(): Pose {
  const pts: Pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
  pts[LM.L_SHOULDER] = { x: 100, y: 100 };
  pts[LM.R_SHOULDER] = { x: 160, y: 100 };
  return pts;
}

describe("offsets", () => {
  it("measures the reachable range from the centre, not the frame count", () => {
    expect(offsetRange(window({ before: 3, after: 5 }))).toEqual({ lo: -3, hi: 5 });
  });

  it("clamps a step past the edge to the edge", () => {
    const win = window({ before: 2, after: 2 });
    expect(clampOffset(win, 7)).toBe(2);
    expect(clampOffset(win, -7)).toBe(-2);
  });

  it("resolves offset 0 to the detected frame however the window was sliced", () => {
    const win = window({ before: 3, after: 9 });
    expect(frameAt(win, 0)).toBe(win.frames[3]);
    expect(frameAt(win, 2)).toBe(win.frames[5]);
    expect(frameAt(win, -3)).toBe(win.frames[0]);
  });
});

describe("shouldWiden", () => {
  it("stays put while the step lands inside the window", () => {
    expect(shouldWiden(window(), 0, 1)).toBe(false);
    expect(shouldWiden(window(), NUDGE_HALF - 1, 1)).toBe(false);
  });

  it("widens when a step reaches past the edge and there is clip left", () => {
    expect(shouldWiden(window(), NUDGE_HALF, 1)).toBe(true);
    expect(shouldWiden(window(), -NUDGE_HALF, -1)).toBe(true);
  });

  it("does not widen where the clip itself ran out", () => {
    // Truncated at the start: fewer frames before the centre than asked for, so
    // decoding a wider window would return exactly these frames again.
    const atClipStart = window({ before: 4 });
    expect(shouldWiden(atClipStart, -4, -1)).toBe(false);
    // The same window still has room going forward.
    expect(shouldWiden(atClipStart, NUDGE_HALF, 1)).toBe(true);
  });

  it("stops widening at the ceiling", () => {
    const maxed = window({ half: NUDGE_HALF_MAX });
    expect(shouldWiden(maxed, NUDGE_HALF_MAX, 1)).toBe(false);
    expect(nextHalf(window())).toBe(Math.min(NUDGE_HALF * 2, NUDGE_HALF_MAX));
    expect(nextHalf(maxed)).toBe(NUDGE_HALF_MAX);
  });
});

describe("halfFor", () => {
  it("uses the default width for a position sitting at or near the detected frame", () => {
    expect(halfFor(0)).toBe(NUDGE_HALF);
    expect(halfFor(-NUDGE_HALF)).toBe(NUDGE_HALF);
  });

  it("re-decodes wide enough to still hold a far-out persisted nudge", () => {
    // Left at 25 frames out, the window dropped, then the position revisited:
    // a default-width window could not reach back to 25 and the next step would
    // clamp to 16, silently pulling the frame back towards the detected one.
    const half = halfFor(25);
    expect(half).toBeGreaterThanOrEqual(25);
    expect(clampOffset(window({ half }), 25)).toBe(25);
    expect(halfFor(-25)).toBe(half);
  });

  it("never exceeds the ceiling, which bounds how far a nudge can be", () => {
    expect(halfFor(NUDGE_HALF_MAX)).toBe(NUDGE_HALF_MAX);
    expect(halfFor(999)).toBe(NUDGE_HALF_MAX);
  });
});

describe("setNudge", () => {
  it("records the stepped frame against its position", () => {
    const map = setNudge({}, "top", frame(2.5), 3);
    expect(map.top?.offset).toBe(3);
    expect(map.top?.timeSec).toBe(2.5);
    expect(offsetOf(map, "top")).toBe(3);
    expect(offsetOf(map, "impact")).toBe(0);
  });

  it("clears the nudge on the way back to the detected frame", () => {
    const map = setNudge(setNudge({}, "top", frame(2.5), 3), "top", frame(2.4), 0);
    expect(map.top).toBeUndefined();
    expect(anyNudged({ ...emptyNudges(), yours: map })).toBe(false);
  });

  it("keeps positions independent of each other", () => {
    let map = setNudge({}, "top", frame(2.5), 3);
    map = setNudge(map, "impact", frame(3.1), -2);
    expect(offsetOf(map, "top")).toBe(3);
    expect(offsetOf(map, "impact")).toBe(-2);
  });
});

describe("posesInPlay", () => {
  const detected = Object.fromEntries(
    EVENTS.map((n) => [n, pose()]),
  ) as Record<EventName, Pose | null>;

  it("passes the detected poses through untouched when nothing is nudged", () => {
    expect(posesInPlay(detected, {})).toEqual(detected);
  });

  it("substitutes the nudged frame's pose, and only at that position", () => {
    const stepped = pose();
    const played = posesInPlay(detected, {
      top: { ...frame(2.5, stepped), offset: 2 },
    });
    expect(played.top).toBe(stepped);
    expect(played.impact).toBe(detected.impact);
  });

  it("scores a nudged frame with no pose as unscorable, not as the old frame", () => {
    const played = posesInPlay(detected, {
      top: { ...frame(2.5, null), offset: 2 },
    });
    expect(played.top).toBeNull();
  });
});
