import { describe, it, expect } from "vitest";
import { LM, type Pose } from "../core/constants";
import { poseAnchor, resolveCrops, videoAnchor } from "./crop";

/** A 33-point pose with the landmarks the crop reads placed explicitly. */
function standingPose({
  shoulderY,
  ankleY,
  hipX,
}: {
  shoulderY: number;
  ankleY: number;
  hipX: number;
}): Pose {
  const pose: Pose = Array.from({ length: 33 }, () => ({ x: hipX, y: 0 }));
  pose[LM.L_SHOULDER] = { x: hipX - 40, y: shoulderY };
  pose[LM.R_SHOULDER] = { x: hipX + 40, y: shoulderY };
  pose[LM.L_HIP] = { x: hipX - 30, y: (shoulderY + ankleY) / 2 };
  pose[LM.R_HIP] = { x: hipX + 30, y: (shoulderY + ankleY) / 2 };
  pose[LM.L_ANKLE] = { x: hipX - 30, y: ankleY };
  pose[LM.R_ANKLE] = { x: hipX + 30, y: ankleY };
  return pose;
}

describe("poseAnchor", () => {
  it("reads span, feet line, and centre from the body", () => {
    const anchor = poseAnchor(
      standingPose({ shoulderY: 400, ankleY: 900, hipX: 500 }),
      1920,
    )!;
    expect(anchor.span).toBeCloseTo(500, 6);
    expect(anchor.feetY).toBeCloseTo(900 + 0.06 * 500, 6);
    expect(anchor.cx).toBeCloseTo(500, 6);
  });

  it("rejects a tiny or disordered pose", () => {
    const tiny = standingPose({ shoulderY: 890, ankleY: 900, hipX: 500 });
    expect(poseAnchor(tiny, 1920)).toBeNull();
    const garbage = standingPose({ shoulderY: 400, ankleY: 900, hipX: 500 });
    garbage[LM.L_HIP].y = 300; // hips above shoulders: not a standing human
    garbage[LM.R_HIP].y = 300;
    expect(poseAnchor(garbage, 1920)).toBeNull();
  });
});

describe("videoAnchor", () => {
  it("takes the median so one bad pose cannot move the crop", () => {
    const good = standingPose({ shoulderY: 400, ankleY: 900, hipX: 500 });
    const drifted = standingPose({ shoulderY: 410, ankleY: 902, hipX: 505 });
    const wild = standingPose({ shoulderY: 700, ankleY: 1900, hipX: 900 });
    const anchor = videoAnchor([good, drifted, wild, null], 1920)!;
    expect(anchor.span).toBeCloseTo(500, 6);
    expect(anchor.cx).toBeCloseTo(505, 6);
  });

  it("is null when no pose is usable", () => {
    expect(videoAnchor([null, null], 1920)).toBeNull();
  });
});

describe("resolveCrops", () => {
  const anchor = poseAnchor(
    standingPose({ shoulderY: 400, ankleY: 900, hipX: 500 }),
    1920,
  )!;

  it("anchors the vertical window on the feet line", () => {
    const [crop] = resolveCrops([{ anchor, frameWidth: 1080 }]);
    expect(crop!.sh).toBeCloseTo(2.0 * 500, 6);
    expect((anchor.feetY - crop!.sy) / crop!.sh).toBeCloseTo(0.84, 6);
  });

  it("shares one window shape across both videos", () => {
    const zoomed = poseAnchor(
      standingPose({ shoulderY: 100, ankleY: 900, hipX: 250 }),
      1000,
    )!;
    // The zoomed clip is only 900 wide against a 1600-tall window, so both
    // sides narrow to the same 900/1600 aspect instead of letterboxing it.
    const [a, b] = resolveCrops([
      { anchor, frameWidth: 1080 },
      { anchor: zoomed, frameWidth: 900 },
    ]);
    expect(a!.sw / a!.sh).toBeCloseTo(b!.sw / b!.sh, 6);
    expect(b!.sw).toBeLessThanOrEqual(900 + 1e-6);
  });

  it("clamps sideways instead of letterboxing near an edge", () => {
    const nearEdge = poseAnchor(
      standingPose({ shoulderY: 400, ankleY: 900, hipX: 80 }),
      1920,
    )!;
    const [crop] = resolveCrops([{ anchor: nearEdge, frameWidth: 1080 }]);
    expect(crop!.sx).toBe(0);
  });

  it("never narrows below the minimum aspect", () => {
    const [a] = resolveCrops([{ anchor, frameWidth: 100 }]);
    expect(a!.sw / a!.sh).toBeCloseTo(0.5, 6);
  });
});
