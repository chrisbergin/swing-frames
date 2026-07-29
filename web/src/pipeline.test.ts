import { describe, it, expect } from "vitest";
import { dropRate, framesToRetain, MAX_DROP_RATE } from "./pipeline";
import { EVENTS, type EventFrames } from "./core/constants";

const events: EventFrames = {
  address: 0,
  toe_up: 35,
  mid_backswing: 43,
  top: 52,
  mid_downswing: 60,
  impact: 62,
  mid_follow_through: 65,
  finish: 72,
};

describe("framesToRetain", () => {
  it("keeps every displayed position", () => {
    const keep = framesToRetain(events, 58, 66);
    for (const name of EVENTS) expect(keep.has(events[name])).toBe(true);
  });

  it("keeps the whole impact window, since refinement can land anywhere in it", () => {
    const keep = framesToRetain(events, 58, 66);
    for (let i = 58; i < 66; i++) expect(keep.has(i)).toBe(true);
    expect(keep.has(57)).toBe(false);
    expect(keep.has(66)).toBe(false);
  });

  it("counts frames that are both a position and in the window only once", () => {
    // impact 62, mid_downswing 60 and mid_follow_through 65 all sit inside
    // the window, so the total is the 8 positions plus the window minus those.
    const keep = framesToRetain(events, 58, 66);
    expect(keep.size).toBe(8 + 8 - 3);
  });
});

describe("dropRate", () => {
  it("measures drops against every frame the decoder saw", () => {
    expect(dropRate(90, 10)).toBeCloseTo(0.1, 10);
    expect(dropRate(100, 0)).toBe(0);
  });

  it("handles a walk that produced nothing", () => {
    expect(dropRate(0, 0)).toBe(0);
  });

  it("rejects the run that had every position wrong", () => {
    // The real numbers from the decode bug: 21 frames through, 36 lost. It
    // still returned 8 confident tiles, at a misread 15fps.
    expect(dropRate(21, 36)).toBeGreaterThan(MAX_DROP_RATE);
  });

  it("tolerates the occasional lost frame, which the track interpolates", () => {
    expect(dropRate(200, 2)).toBeLessThan(MAX_DROP_RATE);
  });
});
