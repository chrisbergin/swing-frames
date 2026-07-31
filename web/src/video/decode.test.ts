import { describe, it, expect } from "vitest";
import { fitScale, rotatedSize } from "./decode";

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

describe("rotatedSize", () => {
  it("keeps dimensions for 0 and 180", () => {
    expect(rotatedSize(1920, 1080, 0)).toEqual({ width: 1920, height: 1080 });
    expect(rotatedSize(1920, 1080, 180)).toEqual({ width: 1920, height: 1080 });
  });

  it("swaps dimensions for 90 and 270", () => {
    expect(rotatedSize(1920, 1080, 90)).toEqual({ width: 1080, height: 1920 });
    expect(rotatedSize(1920, 1080, 270)).toEqual({ width: 1080, height: 1920 });
  });

  it("applies scale before swapping", () => {
    expect(rotatedSize(1920, 1080, 90, 0.5)).toEqual({
      width: 540,
      height: 960,
    });
  });
});
