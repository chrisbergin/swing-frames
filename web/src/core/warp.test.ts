import { describe, expect, it } from "vitest";
import { EVENTS, type EventName } from "./constants";
import {
  anchorMarks,
  anchorPhases,
  anchorTimes,
  planPhases,
  timeAtPhase,
} from "./warp";

/** Event times from a list in swing order. */
function times(list: number[]): Record<EventName, number> {
  return Object.fromEntries(EVENTS.map((n, i) => [n, list[i]])) as Record<
    EventName,
    number
  >;
}

/**
 * A swing with a slow backswing and a fast downswing, which is the shape that
 * makes the two timing modes differ.
 */
const SLOW = [0, 0.5, 1.0, 1.5, 1.65, 1.8, 2.0, 2.4];
/** The same swing shape, quicker overall and with a different tempo split. */
const QUICK = [0, 0.3, 0.55, 0.8, 0.9, 1.0, 1.2, 1.5];

describe("anchorTimes", () => {
  it("reads the 8 positions out in swing order", () => {
    expect(anchorTimes(times(SLOW))).toEqual(SLOW);
  });
});

describe("anchorPhases", () => {
  it("spreads anchors evenly under equal timing, whatever the real tempo", () => {
    const phases = anchorPhases(SLOW, "equal");
    expect(phases[0]).toBe(0);
    expect(phases[7]).toBe(1);
    expect(phases[1]).toBeCloseTo(1 / 7, 10);
    expect(phases).toEqual(anchorPhases(QUICK, "equal"));
  });

  it("places anchors at their real fraction of the clip under your timing", () => {
    const phases = anchorPhases(SLOW, "yours");
    expect(phases[0]).toBe(0);
    expect(phases[7]).toBe(1);
    // Top at 1.5s of a 2.4s swing.
    expect(phases[3]).toBeCloseTo(1.5 / 2.4, 10);
  });

  it("pins the ends and never runs backwards on out-of-order detections", () => {
    const jumbled = [0, 0.5, 0.4, 1.5, 1.65, 1.8, 2.0, 2.4];
    const phases = anchorPhases(jumbled, "yours");
    expect(phases[0]).toBe(0);
    expect(phases[7]).toBe(1);
    for (let i = 1; i < phases.length; i++) {
      expect(phases[i]).toBeGreaterThanOrEqual(phases[i - 1]);
    }
  });

  it("falls back to even spacing for a zero-length swing", () => {
    const flat = new Array(8).fill(2);
    expect(anchorPhases(flat, "yours")).toEqual(anchorPhases(flat, "equal"));
  });
});

describe("timeAtPhase", () => {
  it("lands exactly on each anchor at that anchor's phase", () => {
    const phases = anchorPhases(SLOW, "equal");
    SLOW.forEach((t, i) => {
      expect(timeAtPhase(SLOW, phases, phases[i])).toBeCloseTo(t, 10);
    });
  });

  it("puts both clips on their own anchor at the same phase, which is the warp", () => {
    const phases = anchorPhases(SLOW, "yours");
    // Phase of the top: each clip should report ITS top, not the same clock time.
    const atTop = phases[3];
    expect(timeAtPhase(SLOW, phases, atTop)).toBeCloseTo(1.5, 10);
    expect(timeAtPhase(QUICK, phases, atTop)).toBeCloseTo(0.8, 10);
  });

  it("advances the master clip linearly under your timing", () => {
    const phases = anchorPhases(SLOW, "yours");
    // Linear in phase means the halfway phase is the halfway time.
    expect(timeAtPhase(SLOW, phases, 0.5)).toBeCloseTo(1.2, 6);
  });

  it("stretches the fast downswing under equal timing", () => {
    const phases = anchorPhases(SLOW, "equal");
    // Top is anchor 3 and impact anchor 5, so the downswing spans 2/7 of the
    // scrub while lasting 0.3s of a 2.4s swing.
    const scrubShare = phases[5] - phases[3];
    const timeShare = (SLOW[5] - SLOW[3]) / (SLOW[7] - SLOW[0]);
    expect(scrubShare).toBeCloseTo(2 / 7, 10);
    expect(scrubShare).toBeGreaterThan(timeShare * 2);
  });

  it("is monotonic across the whole scrub for both clips", () => {
    const phases = anchorPhases(SLOW, "yours");
    for (const clip of [SLOW, QUICK]) {
      let prev = -Infinity;
      for (let i = 0; i <= 40; i++) {
        const t = timeAtPhase(clip, phases, i / 40);
        expect(t).toBeGreaterThanOrEqual(prev);
        prev = t;
      }
    }
  });

  it("clamps out-of-range phase to the ends", () => {
    const phases = anchorPhases(SLOW, "equal");
    expect(timeAtPhase(SLOW, phases, -1)).toBeCloseTo(SLOW[0], 10);
    expect(timeAtPhase(SLOW, phases, 2)).toBeCloseTo(SLOW[7], 10);
  });

  it("does not divide by zero when two anchors share a phase", () => {
    const dup = [0, 0, 0, 1.5, 1.65, 1.8, 2.0, 2.4];
    const phases = anchorPhases(dup, "yours");
    const t = timeAtPhase(dup, phases, 0);
    expect(Number.isFinite(t)).toBe(true);
  });
});

describe("planPhases", () => {
  it("spans address to finish inclusive", () => {
    const p = planPhases(5);
    expect(p).toEqual([0, 0.25, 0.5, 0.75, 1]);
  });

  it("degenerates safely", () => {
    expect(planPhases(1)).toEqual([0]);
  });
});

describe("anchorMarks", () => {
  it("pairs every position with its phase, in swing order", () => {
    const marks = anchorMarks(anchorPhases(SLOW, "yours"));
    expect(marks).toHaveLength(EVENTS.length);
    expect(marks[0]).toEqual({ name: "address", phase: 0 });
    expect(marks[7]).toEqual({ name: "finish", phase: 1 });
  });
});
