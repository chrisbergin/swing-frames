import { describe, it, expect } from "vitest";
import { LM, type Pose } from "../core/constants";
import { golferCrop } from "./crop";

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

describe("golferCrop", () => {
  const pose = standingPose({ shoulderY: 400, ankleY: 900, hipX: 500 });
  const crop = golferCrop(pose, 1920)!;

  it("sizes the crop from the shoulder-to-ankle span", () => {
    expect(crop.sh).toBeCloseTo(2.0 * 500, 6);
    expect(crop.sw / crop.sh).toBeCloseTo(0.75, 6);
  });

  it("puts the feet on the standard line", () => {
    const feetY = 900 + 0.06 * 500;
    expect((feetY - crop.sy) / crop.sh).toBeCloseTo(0.84, 6);
  });

  it("centres on the hips", () => {
    expect(crop.sx + crop.sw / 2).toBeCloseTo(500, 6);
  });

  it("scales with distance: half the span means half the crop", () => {
    const far = standingPose({ shoulderY: 650, ankleY: 900, hipX: 500 });
    const farCrop = golferCrop(far, 1920)!;
    expect(farCrop.sh).toBeCloseTo(crop.sh / 2, 6);
  });

  it("rejects a degenerate pose", () => {
    const flat = standingPose({ shoulderY: 900, ankleY: 900, hipX: 500 });
    expect(golferCrop(flat, 1920)).toBeNull();
    // A tiny golfer (under 5% of frame height) cannot anchor a crop either.
    const tiny = standingPose({ shoulderY: 890, ankleY: 900, hipX: 500 });
    expect(golferCrop(tiny, 1920)).toBeNull();
  });
});
