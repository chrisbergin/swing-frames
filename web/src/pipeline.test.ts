import { describe, it, expect } from "vitest";
import {
  DEFAULT_COARSE_SAMPLES,
  dropRate,
  MAX_DROP_RATE,
  retryPlaybackRate,
  sampleIntervalSeconds,
} from "./pipeline";

describe("sampleIntervalSeconds", () => {
  it("spreads the samples evenly across the clip", () => {
    expect(sampleIntervalSeconds(4.8, 48)).toBeCloseTo(0.1, 10);
    // A long clip is sampled more sparsely, which is the whole point: cost
    // tracks the sample count, not the length of the video.
    expect(sampleIntervalSeconds(60, 48)).toBeCloseTo(1.25, 10);
  });

  it("keeps a swing's worth of samples well above the episode threshold", () => {
    // Detection needs at least 3 samples inside a hands-high episode, and
    // there are two of those in a swing, so the sample count has to leave room.
    expect(DEFAULT_COARSE_SAMPLES).toBeGreaterThan(3 * 2 * 4);
  });

  it("disables sampling for a degenerate clip rather than dividing by zero", () => {
    expect(sampleIntervalSeconds(0, 48)).toBe(0);
    expect(sampleIntervalSeconds(NaN, 48)).toBe(0);
    expect(sampleIntervalSeconds(4, 0)).toBe(0);
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

describe("retryPlaybackRate", () => {
  it("slows enough that a frame's work fits in a frame interval", () => {
    // 30fps is a frame every 33ms; 100ms of work per frame needs roughly a
    // quarter speed to fit with headroom to spare.
    const rate = retryPlaybackRate(100, 30, 1);
    expect(rate).toBeCloseTo((1000 / 30) * 0.7 / 100, 6);
    expect(1000 / 30 / rate).toBeGreaterThan(100);
  });

  it("always ends up slower than the attempt that just failed", () => {
    // Cheap frames that still dropped: something other than raw speed is
    // wrong, so back off anyway rather than retrying at the same rate.
    expect(retryPlaybackRate(1, 30, 1)).toBeLessThan(1);
    expect(retryPlaybackRate(1, 30, 0.5)).toBeLessThan(0.5);
  });

  it("refuses to slow past the point of being worth waiting for", () => {
    expect(retryPlaybackRate(100000, 240, 1)).toBe(0.05);
  });

  it("survives a degenerate first attempt", () => {
    expect(retryPlaybackRate(0, 30, 1)).toBe(0.05);
    expect(retryPlaybackRate(50, 0, 1)).toBe(0.05);
  });
});
