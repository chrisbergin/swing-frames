import { describe, it, expect } from "vitest";
import { countDrops, estimateFps, fitScale } from "./decode";

/** Presentation timestamps for `count` frames at a steady `fps`. */
function steadyTimes(fps: number, count: number): number[] {
  return Array.from({ length: count }, (_, i) => i / fps);
}

describe("estimateFps", () => {
  it("recovers a steady frame rate from timestamps", () => {
    expect(estimateFps(steadyTimes(30, 40))).toBeCloseTo(30, 6);
    expect(estimateFps(steadyTimes(240, 100))).toBeCloseTo(240, 6);
  });

  it("is not thrown off by a long gap from dropped frames", () => {
    // A 10 second stall in the middle would wreck a mean-based estimate.
    const times = [...steadyTimes(30, 3), 10, 10 + 1 / 30];
    expect(estimateFps(times)).toBeCloseTo(30, 6);
  });

  it("ignores non-advancing timestamps", () => {
    // A repeated frame contributes no interval rather than a zero one.
    const times = [0, 1 / 30, 1 / 30, 2 / 30, 3 / 30];
    expect(estimateFps(times)).toBeCloseTo(30, 6);
  });

  it("falls back to 30 when there is nothing to measure", () => {
    expect(estimateFps([])).toBe(30);
    expect(estimateFps([0])).toBe(30);
    expect(estimateFps([5, 5, 5])).toBe(30);
  });
});

describe("fitScale", () => {
  it("shrinks a large frame to the long side", () => {
    expect(fitScale(1920, 1080, 960)).toBeCloseTo(0.5, 10);
    // Portrait phone video: height is the long side.
    expect(fitScale(1080, 1920, 960)).toBeCloseTo(0.5, 10);
  });

  it("never enlarges a frame that already fits", () => {
    expect(fitScale(640, 360, 960)).toBe(1);
    expect(fitScale(960, 540, 960)).toBe(1);
  });
});

describe("countDrops", () => {
  it("counts nothing when every frame arrives", () => {
    expect(countDrops([1, 2, 3, 4, 5])).toBe(0);
  });

  it("counts the frames missing from a gap", () => {
    // 2 and 3 never arrived.
    expect(countDrops([1, 4, 5])).toBe(2);
  });

  it("adds up several gaps", () => {
    expect(countDrops([1, 3, 4, 8])).toBe(1 + 3);
  });

  it("handles a walk with no frames", () => {
    expect(countDrops([])).toBe(0);
    expect(countDrops([7])).toBe(0);
  });
});
