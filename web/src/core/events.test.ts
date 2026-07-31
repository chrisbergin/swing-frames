import { describe, it, expect } from "vitest";
import {
  crossing,
  detectEvents,
  impactRefineWindow,
  percentile,
  selectImpactFrame,
  smooth,
  smoothWindow,
  SwingDetectionError,
  wristTrack,
} from "./events";
import { EVENTS, LM, type EventFrames, type Pose } from "./constants";

/** `count` samples from `from` to `to` inclusive. */
function ramp(from: number, to: number, count: number): number[] {
  return Array.from(
    { length: count },
    (_, k) => from + ((to - from) * k) / (count - 1),
  );
}

function hold(value: number, count: number): number[] {
  return new Array<number>(count).fill(value);
}

/** A pose whose wrist midpoint sits at (x, y). Detection reads nothing else. */
function wristPose(y: number, x = 50): Pose {
  const pts: Pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
  pts[LM.L_WRIST] = { x: x - 5, y };
  pts[LM.R_WRIST] = { x: x + 5, y };
  return pts;
}

/** Frame indices in swing order, for ordering assertions. */
function inOrder(events: EventFrames): number[] {
  return EVENTS.map((e) => events[e]);
}

function isStrictlyIncreasing(xs: number[]): boolean {
  return xs.every((v, i) => i === 0 || v > xs[i - 1]);
}

describe("smoothWindow", () => {
  it("scales with frame rate but never goes below 3", () => {
    expect(smoothWindow(15)).toBe(3); // trunc(1) floored up to 3
    expect(smoothWindow(30)).toBe(3); // trunc(2) floored up to 3
    expect(smoothWindow(60)).toBe(4);
    expect(smoothWindow(240)).toBe(16);
  });
});

describe("smooth", () => {
  it("leaves a constant signal unchanged", () => {
    expect(smooth(hold(7, 10), 5)).toEqual(hold(7, 10));
  });

  it("extends the edges rather than sagging toward zero", () => {
    const out = smooth([1, 2, 3, 4, 5], 3);
    expect(out).toHaveLength(5);
    expect(out[0]).toBeCloseTo((1 + 1 + 2) / 3, 10);
    expect(out[2]).toBeCloseTo(3, 10);
    expect(out[4]).toBeCloseTo((4 + 5 + 5) / 3, 10);
  });

  it("forces the window odd and at least 3", () => {
    // window 2 and window 3 both resolve to 3.
    expect(smooth([1, 2, 3, 4, 5], 2)).toEqual(smooth([1, 2, 3, 4, 5], 3));
  });

  it("returns a linear signal unchanged in the interior", () => {
    const line = Array.from({ length: 20 }, (_, i) => 100 + 10 * i);
    const out = smooth(line, 3);
    for (let i = 1; i < 19; i++) expect(out[i]).toBeCloseTo(line[i], 10);
  });

  it("handles an empty signal", () => {
    expect(smooth([], 5)).toEqual([]);
  });
});

describe("percentile", () => {
  it("interpolates linearly between order statistics", () => {
    expect(percentile([1, 2, 3, 4], 50)).toBeCloseTo(2.5, 10);
  });

  it("returns the extremes at 0 and 100", () => {
    const xs = [5, 1, 9, 3];
    expect(percentile(xs, 0)).toBe(1);
    expect(percentile(xs, 100)).toBe(9);
  });

  it("does not mutate its input", () => {
    const xs = [5, 1, 9, 3];
    percentile(xs, 50);
    expect(xs).toEqual([5, 1, 9, 3]);
  });
});

describe("crossing", () => {
  const rising = [300, 250, 200, 150, 100]; // hands going up (y falling)

  it("finds the first frame at or above a rising level", () => {
    expect(crossing(rising, 0, 5, 200, "up")).toBe(2);
  });

  it("finds the first frame at or below a falling level", () => {
    const falling = [100, 150, 200, 250, 300];
    expect(crossing(falling, 0, 5, 200, "down")).toBe(2);
  });

  it("respects the search bounds", () => {
    expect(crossing(rising, 3, 5, 200, "up")).toBe(3);
  });

  it("returns null when the level is never reached", () => {
    expect(crossing(rising, 0, 5, 50, "up")).toBeNull();
  });

  it("clamps an end index past the signal length", () => {
    expect(crossing(rising, 0, 999, 100, "up")).toBe(4);
  });
});

describe("selectImpactFrame", () => {
  it("takes the last frame within a hair of the lowest hands", () => {
    // y grows downward, so the largest y is the lowest hand position.
    // Frames 11 and 12 are within 5% of the bottom; the later one wins.
    const measured = [
      { frame: 10, y: 180 },
      { frame: 11, y: 200 },
      { frame: 12, y: 199 },
      { frame: 13, y: 150 },
    ];
    expect(selectImpactFrame(measured, 11)).toBe(12);
  });

  it("does not reach past a frame outside the tolerance", () => {
    const measured = [
      { frame: 10, y: 200 },
      { frame: 11, y: 100 },
    ];
    expect(selectImpactFrame(measured, 10)).toBe(10);
  });

  it("falls back when nothing could be measured", () => {
    expect(selectImpactFrame([], 42)).toBe(42);
  });
});

describe("impactRefineWindow", () => {
  const events = { top: 40, impact: 80 } as EventFrames;

  it("searches mostly backwards, since tracked impact is only ever late", () => {
    // fps 30 gives a window of 3: 2*3 back, 3+1 forward.
    const { lo, hi } = impactRefineWindow(events, 30, 200);
    expect(lo).toBe(74);
    expect(hi).toBe(84);
  });

  it("never searches before the top of the backswing", () => {
    const tight = { top: 78, impact: 80 } as EventFrames;
    expect(impactRefineWindow(tight, 30, 200).lo).toBe(78);
  });

  it("never searches past the end of the clip", () => {
    expect(impactRefineWindow(events, 30, 82).hi).toBe(82);
  });
});

describe("wristTrack", () => {
  it("interpolates across frames where pose was lost", () => {
    // A perfectly linear hand path with three frames dropped. Linear
    // interpolation restores them exactly, and a moving average leaves a
    // linear signal untouched in the interior, so the values come back exact.
    const poses: (Pose | null)[] = Array.from({ length: 20 }, (_, i) =>
      wristPose(100 + 10 * i),
    );
    poses[5] = null;
    poses[6] = null;
    poses[12] = null;

    const { ys } = wristTrack(poses, 30);
    expect(ys.some(Number.isNaN)).toBe(false);
    expect(ys[5]).toBeCloseTo(150, 8);
    expect(ys[6]).toBeCloseTo(160, 8);
    expect(ys[12]).toBeCloseTo(220, 8);
    expect(ys[10]).toBeCloseTo(200, 8);
  });

  it("tracks the wrist midpoint, not one wrist", () => {
    const poses = Array.from({ length: 12 }, () => wristPose(100, 60));
    const { xs } = wristTrack(poses, 30);
    expect(xs[5]).toBeCloseTo(60, 8);
  });

  it("refuses a clip where pose was mostly not detected", () => {
    const poses: (Pose | null)[] = Array.from({ length: 20 }, (_, i) =>
      i < 8 ? wristPose(100) : null,
    );
    expect(() => wristTrack(poses, 30)).toThrow(SwingDetectionError);
    expect(() => wristTrack(poses, 30)).toThrow(/only 8 of 20 sampled frames/);
  });
});

describe("detectEvents", () => {
  /** Continuous footage: a phone video of one swing, no cuts. */
  const continuous = [
    ...hold(300, 20), // address
    ...ramp(300, 100, 30), // backswing
    ...hold(100, 10), // top
    ...ramp(100, 305, 20), // downswing through impact
    ...ramp(305, 110, 20), // follow through
    ...hold(110, 20), // finish hold
  ];

  it("orders all 8 positions through a continuous swing", () => {
    const events = detectEvents(continuous, 30);
    const frames = inOrder(events);
    expect(frames).toHaveLength(8);
    expect(isStrictlyIncreasing(frames)).toBe(true);
  });

  it("anchors address, top, impact and finish in the right regions", () => {
    const events = detectEvents(continuous, 30);
    expect(events.address).toBe(0); // clip starts with hands already set
    expect(events.top).toBeGreaterThanOrEqual(45); // top hold is frames 50-59
    expect(events.top).toBeLessThanOrEqual(60);
    expect(events.impact).toBeGreaterThanOrEqual(75); // hands lowest near 79
    expect(events.impact).toBeLessThanOrEqual(85);
    expect(events.finish).toBeGreaterThanOrEqual(95);
  });

  /**
   * Pause-and-step analysis footage: freeze frames joined by jumps. This is
   * the shape that broke the old speed-based detection, since speed is zero
   * on every freeze and spikes at every cut.
   */
  const stepFrame = [
    ...hold(300, 10), // address freeze
    ...hold(240, 8), // toe up freeze
    ...hold(160, 8), // mid backswing freeze
    ...hold(100, 10), // top freeze
    ...hold(200, 8), // mid downswing freeze
    ...hold(305, 10), // impact freeze
    ...hold(200, 8), // mid follow through freeze
    ...hold(110, 12), // finish freeze
  ];

  it("orders all 8 positions through step-frame footage", () => {
    const events = detectEvents(stepFrame, 30);
    expect(isStrictlyIncreasing(inOrder(events))).toBe(true);
    expect(events.address).toBe(0);
    expect(events.top).toBe(26); // first frame of the top freeze
    expect(events.impact).toBe(44); // first frame of the impact freeze
  });

  it("falls back to the neighbour midpoint for a crossing never reached", () => {
    // The mid-downswing freeze sits at 200, just shy of the 202.5 level the
    // crossing looks for, so that position resolves from its neighbours.
    const events = detectEvents(stepFrame, 30);
    expect(events.mid_downswing).toBe(
      Math.floor((events.top + events.impact) / 2),
    );
    expect(events.mid_downswing).toBeGreaterThan(events.top);
    expect(events.mid_downswing).toBeLessThan(events.impact);
  });

  it("handles a clip cut short before the finish hold", () => {
    const truncated = [
      ...hold(300, 10),
      ...ramp(300, 100, 20),
      ...hold(100, 8),
      ...ramp(100, 310, 15), // ends at impact, no follow through
    ];
    const events = detectEvents(truncated, 30);
    // With no second hands-high episode, the finish falls on the last frame,
    // which is also where impact lands because the clip stops there.
    expect(events.finish).toBe(truncated.length - 1);
    expect(events.impact).toBe(truncated.length - 1);
    expect(inOrder(events).every((v, i, a) => i === 0 || v >= a[i - 1])).toBe(true);
  });

  it("backs the address off the takeaway when the clip starts early", () => {
    // Hands start slightly raised and only settle to address a few frames in,
    // so frame 0 is not a valid address and the late-baseline branch runs.
    // The opening stays well below the hands-high threshold, or it would read
    // as a backswing top (see the trimming test below).
    const lateAddress = [
      ...hold(250, 6), // clip opens with hands a little off the ball
      ...ramp(250, 300, 8), // settling down into address
      ...hold(300, 20), // address hold
      ...ramp(300, 100, 30),
      ...hold(100, 10),
      ...ramp(100, 305, 20),
      ...ramp(305, 110, 20),
      ...hold(110, 20),
    ];
    const events = detectEvents(lateAddress, 30);
    expect(events.address).toBeGreaterThan(0);
    expect(events.address).toBeLessThan(events.top);
    // Backed off the takeaway boundary, so it lands inside the address hold.
    expect(events.address).toBeGreaterThanOrEqual(14);
    expect(events.address).toBeLessThanOrEqual(33);
    expect(isStrictlyIncreasing(inOrder(events))).toBe(true);
  });

  it("reads a clip that opens with the hands already high as starting at the top", () => {
    // Documented limitation, not a bug: the first hands-high episode is taken
    // as the backswing top, so an untrimmed clip that opens mid-swing (or on a
    // practice swing) anchors everything wrongly. This is why the recording
    // guidance insists on trimming to a single swing from address.
    const untrimmed = [
      ...hold(150, 6), // clip opens mid-motion, hands already up
      ...ramp(150, 300, 10),
      ...hold(300, 20),
      ...ramp(300, 100, 30),
      ...hold(100, 10),
      ...ramp(100, 305, 20),
      ...ramp(305, 110, 20),
      ...hold(110, 20),
    ];
    const events = detectEvents(untrimmed, 30);
    expect(events.top).toBe(0);
    expect(events.address).toBe(0);
  });

  it("rejects a clip with no backswing motion", () => {
    expect(() => detectEvents(hold(300, 50), 30)).toThrow(SwingDetectionError);
    expect(() => detectEvents(hold(300, 50), 30)).toThrow(/trimmed to one swing/);
  });

  it("ignores hands-high blips shorter than 3 frames", () => {
    // A 2-frame spike of tracker noise before the real backswing must not
    // be mistaken for the top of the swing.
    const withBlip = [...continuous];
    withBlip[5] = 90;
    withBlip[6] = 90;
    const events = detectEvents(withBlip, 30);
    expect(events.top).toBeGreaterThan(40);
  });
});
