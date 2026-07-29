import { describe, it, expect } from "vitest";
import {
  ANGLE_NAMES,
  biggestGaps,
  compareSwings,
  jointAngles,
  overallScore,
  type Similarity,
} from "./angles";
import { EVENTS, LM, type EventFrames, type Pose } from "./constants";

/**
 * An anatomically plausible figure facing the camera, in screen pixels
 * (y grows downward). Arms and legs are straight, so every limb angle reads
 * 180, and the shoulders sit directly above the hips, so spine tilt reads 0.
 * Tests bend one joint at a time from this baseline.
 */
function basePose(overrides: Partial<Record<number, [number, number]>> = {}): Pose {
  const pts: Pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
  const set = (i: number, x: number, y: number) => {
    pts[i] = { x, y };
  };
  set(LM.L_SHOULDER, 40, 100);
  set(LM.R_SHOULDER, 60, 100);
  set(LM.L_ELBOW, 40, 130);
  set(LM.R_ELBOW, 60, 130);
  set(LM.L_WRIST, 40, 160);
  set(LM.R_WRIST, 60, 160);
  set(LM.L_HIP, 42, 170);
  set(LM.R_HIP, 58, 170);
  set(LM.L_KNEE, 42, 210);
  set(LM.R_KNEE, 58, 210);
  set(LM.L_ANKLE, 42, 250);
  set(LM.R_ANKLE, 58, 250);
  for (const [i, xy] of Object.entries(overrides)) {
    if (!xy) continue;
    set(Number(i), xy[0], xy[1]);
  }
  return pts;
}

/** Every position pointing at the same frame index, for single-pose comparisons. */
function allAt(frame: number): EventFrames {
  return Object.fromEntries(EVENTS.map((e) => [e, frame])) as EventFrames;
}

describe("jointAngles", () => {
  it("reads a straight limb as 180 degrees", () => {
    const a = jointAngles(basePose());
    expect(a.l_elbow).toBeCloseTo(180, 6);
    expect(a.r_elbow).toBeCloseTo(180, 6);
    expect(a.l_knee).toBeCloseTo(180, 6);
    expect(a.r_knee).toBeCloseTo(180, 6);
  });

  it("reads a perpendicular bend as 90 degrees", () => {
    // Forearm swings out horizontally from the elbow.
    const a = jointAngles(basePose({ [LM.R_WRIST]: [90, 130] }));
    expect(a.r_elbow).toBeCloseTo(90, 6);
  });

  it("reads shoulders stacked over hips as zero spine tilt", () => {
    expect(jointAngles(basePose()).spine_tilt).toBeCloseTo(0, 6);
  });

  it("signs spine tilt by lean direction", () => {
    // Shoulders shifted to larger x (screen right) relative to the hips.
    const right = jointAngles(
      basePose({ [LM.L_SHOULDER]: [110, 100], [LM.R_SHOULDER]: [130, 100] }),
    );
    const left = jointAngles(
      basePose({ [LM.L_SHOULDER]: [-30, 100], [LM.R_SHOULDER]: [-10, 100] }),
    );
    expect(right.spine_tilt).toBeGreaterThan(0);
    expect(left.spine_tilt).toBeLessThan(0);
    expect(right.spine_tilt).toBeCloseTo(-left.spine_tilt, 6);
  });

  it("is invariant to scale and translation", () => {
    const pts = basePose({ [LM.R_WRIST]: [90, 130] });
    const moved = pts.map((p) => ({ x: p.x * 3 + 500, y: p.y * 3 - 250 }));
    const a = jointAngles(pts);
    const b = jointAngles(moved);
    for (const k of ANGLE_NAMES) {
      expect(b[k]).toBeCloseTo(a[k], 6);
    }
  });

  it("does not divide by zero on coincident landmarks", () => {
    // Wrist collapsed onto the elbow: degenerate, but must stay finite.
    const a = jointAngles(basePose({ [LM.R_WRIST]: [60, 130] }));
    expect(Number.isFinite(a.r_elbow)).toBe(true);
  });
});

describe("compareSwings", () => {
  it("scores an identical swing as 100 with zero gaps", () => {
    const poses = [basePose()];
    const sim = compareSwings(poses, allAt(0), poses, allAt(0));
    for (const name of EVENTS) {
      expect(sim[name]).not.toBeNull();
      expect(sim[name]!.score).toBe(100);
      expect(sim[name]!.meanAngleDiffDeg).toBe(0);
      for (const k of ANGLE_NAMES) {
        expect(sim[name]!.jointDiffsDeg[k]).toBe(0);
      }
    }
    expect(overallScore(sim)).toBe(100);
  });

  it("loses 2 points per degree of mean difference", () => {
    // One joint differs by exactly 90 degrees and the other 8 by 0, so the
    // mean across the 9 measures is 10 and the score should be 100 - 2*10.
    const a = [basePose()];
    const b = [basePose({ [LM.R_WRIST]: [90, 130] })];
    const sim = compareSwings(a, allAt(0), b, allAt(0));
    const top = sim.top!;
    expect(top.jointDiffsDeg.r_elbow).toBeCloseTo(90, 6);
    expect(top.jointDiffsDeg.l_elbow).toBe(0);
    expect(top.meanAngleDiffDeg).toBeCloseTo(10, 6);
    expect(top.score).toBeCloseTo(80, 6);
  });

  it("floors the score at zero instead of going negative", () => {
    // Shoulders below the hips inverts the spine and wrecks every limb angle:
    // the raw formula would go negative, so this pins the max(0, ...) clamp.
    const a = [basePose()];
    const b = [
      basePose({
        [LM.L_SHOULDER]: [40, 240],
        [LM.R_SHOULDER]: [60, 240],
        [LM.R_WRIST]: [90, 130],
        [LM.L_WRIST]: [10, 130],
        [LM.L_ANKLE]: [2, 210],
        [LM.R_ANKLE]: [98, 210],
      }),
    ];
    const sim = compareSwings(a, allAt(0), b, allAt(0));
    expect(sim.top!.score).toBe(0);
    expect(sim.top!.meanAngleDiffDeg).toBeGreaterThan(50);
  });

  it("returns null for a position with no pose on either side", () => {
    const a: (Pose | null)[] = [null];
    const b = [basePose()];
    const sim = compareSwings(a, allAt(0), b, allAt(0));
    expect(sim.address).toBeNull();
    expect(overallScore(sim)).toBeNull();
  });

  it("scores only the positions that have poses on both sides", () => {
    // Frame 0 has a pose, frame 1 does not; address points at 0, the rest at 1.
    const a: (Pose | null)[] = [basePose(), null];
    const b: (Pose | null)[] = [basePose(), basePose()];
    const events = { ...allAt(1), address: 0 };
    const sim = compareSwings(a, events, b, events);
    expect(sim.address).not.toBeNull();
    expect(sim.top).toBeNull();
    // Only address scored, and it is a perfect match.
    expect(overallScore(sim)).toBe(100);
  });
});

describe("biggestGaps", () => {
  it("returns the widest joint differences first", () => {
    const sim: Similarity = compareSwings(
      [basePose()],
      allAt(0),
      [basePose({ [LM.R_WRIST]: [90, 130], [LM.L_ANKLE]: [2, 210] })],
      allAt(0),
    );
    const gaps = biggestGaps(sim.top!, 2);
    expect(gaps).toHaveLength(2);
    expect(gaps.map((g) => g[0]).sort()).toEqual(["l_knee", "r_elbow"]);
    expect(gaps[0][1]).toBeGreaterThanOrEqual(gaps[1][1]);
  });
});
