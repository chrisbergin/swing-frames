import { describe, it, expect } from "vitest";
import { firstPose, MODEL_URLS, toPose } from "./landmarker";

describe("toPose", () => {
  it("scales normalized landmarks up to pixels", () => {
    const pose = toPose([{ x: 0.5, y: 0.25 }, { x: 1, y: 1 }], 640, 480);
    expect(pose).toEqual([
      { x: 320, y: 120 },
      { x: 640, y: 480 },
    ]);
  });

  it("returns null when nothing was detected", () => {
    expect(toPose(undefined, 640, 480)).toBeNull();
    expect(toPose([], 640, 480)).toBeNull();
  });
});

describe("firstPose", () => {
  it("takes the first pose and ignores any others", () => {
    const result = {
      landmarks: [[{ x: 0.5, y: 0.5 }], [{ x: 0.1, y: 0.1 }]],
    };
    expect(firstPose(result, 200, 100)).toEqual([{ x: 100, y: 50 }]);
  });

  it("returns null for an empty result", () => {
    expect(firstPose({ landmarks: [] }, 200, 100)).toBeNull();
  });
});

describe("MODEL_URLS", () => {
  it("points at the official model for each size", () => {
    for (const [name, url] of Object.entries(MODEL_URLS)) {
      expect(url).toContain(`pose_landmarker_${name}.task`);
      expect(url.startsWith("https://storage.googleapis.com/")).toBe(true);
    }
  });
});
