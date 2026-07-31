import { describe, it, expect } from "vitest";
import {
  coarseSampleCount,
  DEFAULT_COARSE_SAMPLES,
  planSampleTimes,
  planWindowTimes,
  sampleIntervalSeconds,
} from "./pipeline";

describe("sampleIntervalSeconds", () => {
  it("spreads the samples evenly across the clip", () => {
    expect(sampleIntervalSeconds(4.8, 48)).toBeCloseTo(0.1, 10);
    // A long clip is sampled more sparsely, which is the point: cost tracks
    // the sample count, not the length of the video.
    expect(sampleIntervalSeconds(60, 48)).toBeCloseTo(1.25, 10);
  });

  it("keeps a swing's worth of samples well above the episode threshold", () => {
    // Detection needs at least 3 samples inside a hands-high episode, and there
    // are two of those in a swing, so the count has to leave room.
    expect(DEFAULT_COARSE_SAMPLES).toBeGreaterThan(3 * 2 * 4);
  });

  it("handles a degenerate clip rather than dividing by zero", () => {
    expect(sampleIntervalSeconds(0, 48)).toBe(0);
    expect(sampleIntervalSeconds(NaN, 48)).toBe(0);
    expect(sampleIntervalSeconds(4, 0)).toBe(0);
  });
});

describe("planSampleTimes", () => {
  it("keeps every sample inside the clip", () => {
    const times = planSampleTimes(10, 48);
    expect(times).toHaveLength(48);
    expect(Math.min(...times)).toBeGreaterThan(0);
    expect(Math.max(...times)).toBeLessThan(10);
  });

  it("spaces samples evenly", () => {
    const times = planSampleTimes(10, 5);
    expect(times).toEqual([1, 3, 5, 7, 9]);
  });

  it("still returns something for a zero-length clip", () => {
    expect(planSampleTimes(0, 48)).toEqual([0]);
  });
});

describe("planWindowTimes", () => {
  it("spans the window and centres on the target", () => {
    const times = planWindowTimes(5, 1, 5, 10);
    expect(times).toEqual([4, 4.5, 5, 5.5, 6]);
  });

  it("clamps to the clip rather than seeking off the end", () => {
    const times = planWindowTimes(0.1, 1, 5, 10);
    expect(Math.min(...times)).toBe(0);
    const late = planWindowTimes(9.9, 1, 5, 10);
    expect(Math.max(...late)).toBeLessThanOrEqual(10);
  });

  it("degrades to the centre when asked for a single sample", () => {
    expect(planWindowTimes(5, 1, 1, 10)).toEqual([5]);
  });
});


describe("coarseSampleCount", () => {
  it("aims for a time resolution rather than a fixed count", () => {
    // A 30s analysis clip at a fixed 48 samples lands one every 0.63s, coarse
    // enough to pick the wrong freeze frame.
    expect(coarseSampleCount(30)).toBeGreaterThan(DEFAULT_COARSE_SAMPLES);
    expect(30 / coarseSampleCount(30)).toBeLessThan(0.63);
  });

  it("never drops below the floor on a short clip", () => {
    expect(coarseSampleCount(1)).toBe(DEFAULT_COARSE_SAMPLES);
    expect(coarseSampleCount(4.2)).toBe(DEFAULT_COARSE_SAMPLES);
  });

  it("caps a long clip so cost stays bounded", () => {
    expect(coarseSampleCount(600)).toBe(coarseSampleCount(6000));
    expect(coarseSampleCount(600)).toBeLessThanOrEqual(150);
  });

  it("falls back for a clip with no usable duration", () => {
    expect(coarseSampleCount(0)).toBe(DEFAULT_COARSE_SAMPLES);
    expect(coarseSampleCount(NaN)).toBe(DEFAULT_COARSE_SAMPLES);
  });
});
