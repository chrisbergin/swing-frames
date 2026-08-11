/** Joint-angle measurement and swing-to-swing similarity scoring.
 *
 * Ported from joint_angles() and compare_swings() in swing_frames.py.
 */

import {
  EVENTS,
  LM,
  round1,
  type EventFrames,
  type EventName,
  type Point,
  type Pose,
} from "./constants";

/** The 9 measured quantities: 8 joint angles plus spine tilt. */
export const ANGLE_NAMES = [
  "l_elbow",
  "r_elbow",
  "l_shoulder",
  "r_shoulder",
  "l_hip",
  "r_hip",
  "l_knee",
  "r_knee",
  "spine_tilt",
] as const;

export type AngleName = (typeof ANGLE_NAMES)[number];

export type JointAngles = Record<AngleName, number>;

export interface PositionSimilarity {
  /** 0-100, where 100 is identical joint angles. */
  score: number;
  meanAngleDiffDeg: number;
  jointDiffsDeg: Record<AngleName, number>;
}

export type Similarity = Record<EventName, PositionSimilarity | null>;

const DEG = 180 / Math.PI;

function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/** Interior angle in degrees at vertex `b`, between rays b->a and b->c. */
function angleAt(pts: Pose, a: number, b: number, c: number): number {
  const v1 = { x: pts[a].x - pts[b].x, y: pts[a].y - pts[b].y };
  const v2 = { x: pts[c].x - pts[b].x, y: pts[c].y - pts[b].y };
  const n1 = Math.hypot(v1.x, v1.y);
  const n2 = Math.hypot(v2.x, v2.y);
  // Coincident landmarks have no defined angle; 90 is the answer the Python's
  // old epsilon denominator produced, and both sides now use this explicit
  // guard. An epsilon in the normal path sits where acos is steepest, so a
  // straight limb measured 179.99991 rather than 180 and scale invariance
  // broke at 1e-4 degrees.
  if (n1 === 0 || n2 === 0) return 90;
  const cos = (v1.x * v2.x + v1.y * v2.y) / (n1 * n2);
  return Math.acos(Math.min(1, Math.max(-1, cos))) * DEG;
}

/**
 * 2D angles at the major joints, plus spine tilt from vertical.
 *
 * Angles are size- and distance-invariant, so two bodies filmed at the same
 * camera angle can be compared directly without any scaling or registration.
 */
export function jointAngles(pts: Pose): JointAngles {
  const shoulders = midpoint(pts[LM.L_SHOULDER], pts[LM.R_SHOULDER]);
  const hips = midpoint(pts[LM.L_HIP], pts[LM.R_HIP]);
  // Vector hips -> shoulders. Negate y so that "up the screen" is positive,
  // making a perfectly vertical spine read as 0 and a lean read as its angle.
  const spine = { x: shoulders.x - hips.x, y: shoulders.y - hips.y };

  return {
    l_elbow: angleAt(pts, LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST),
    r_elbow: angleAt(pts, LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST),
    l_shoulder: angleAt(pts, LM.L_ELBOW, LM.L_SHOULDER, LM.L_HIP),
    r_shoulder: angleAt(pts, LM.R_ELBOW, LM.R_SHOULDER, LM.R_HIP),
    l_hip: angleAt(pts, LM.L_SHOULDER, LM.L_HIP, LM.L_KNEE),
    r_hip: angleAt(pts, LM.R_SHOULDER, LM.R_HIP, LM.R_KNEE),
    l_knee: angleAt(pts, LM.L_HIP, LM.L_KNEE, LM.L_ANKLE),
    r_knee: angleAt(pts, LM.R_HIP, LM.R_KNEE, LM.R_ANKLE),
    spine_tilt: Math.atan2(spine.x, -spine.y) * DEG,
  };
}

/**
 * Joint-angle similarity at each matched position.
 *
 * Per position: mean absolute angle difference across the 9 measured angles,
 * mapped to a 0-100 score (100 = identical, 2 points lost per degree of
 * average difference). Only meaningful when both videos are shot from the
 * same camera angle and the golfers share handedness.
 */
export function compareSwings(
  posesA: (Pose | null)[],
  eventsA: EventFrames,
  posesB: (Pose | null)[],
  eventsB: EventFrames,
): Similarity {
  return compareEventPoses(
    Object.fromEntries(
      EVENTS.map((n) => [n, posesA[eventsA[n]] ?? null]),
    ) as Record<EventName, Pose | null>,
    Object.fromEntries(
      EVENTS.map((n) => [n, posesB[eventsB[n]] ?? null]),
    ) as Record<EventName, Pose | null>,
  );
}

/**
 * Similarity from the poses at each position directly.
 *
 * Angles are scale-invariant, so the two sides do not have to be measured at
 * the same resolution: poses taken off downscaled frames compare correctly
 * against poses taken off full-size ones.
 */
export function compareEventPoses(
  posesA: Record<EventName, Pose | null>,
  posesB: Record<EventName, Pose | null>,
): Similarity {
  const result = {} as Similarity;
  for (const name of EVENTS) {
    result[name] = comparePoses(posesA[name], posesB[name]);
  }
  return result;
}

/**
 * Similarity between one pair of poses, or null if either is missing.
 *
 * The unit both the per-position scores and the sync view's divergence curve
 * are built from: the curve scores a pair at every sampled phase point, which
 * is the same measurement applied at moments that are not named positions.
 */
export function comparePoses(
  pa: Pose | null,
  pb: Pose | null,
): PositionSimilarity | null {
  if (!pa || !pb) return null;

  const aa = jointAngles(pa);
  const ab = jointAngles(pb);
  const diffs = {} as Record<AngleName, number>;
  let total = 0;
  for (const k of ANGLE_NAMES) {
    const d = Math.abs(aa[k] - ab[k]);
    diffs[k] = round1(d);
    total += d;
  }
  const meanDiff = total / ANGLE_NAMES.length;

  return {
    score: round1(Math.max(0, 100 - 2 * meanDiff)),
    meanAngleDiffDeg: round1(meanDiff),
    jointDiffsDeg: diffs,
  };
}

/** Mean score across the positions that could be scored, or null if none could. */
export function overallScore(similarity: Similarity): number | null {
  const scored = EVENTS.map((n) => similarity[n]).filter(
    (v): v is PositionSimilarity => v !== null,
  );
  if (scored.length === 0) return null;
  return round1(scored.reduce((a, v) => a + v.score, 0) / scored.length);
}

/** The `n` joints that differ most, biggest first. Drives the on-sheet labels. */
export function biggestGaps(
  position: PositionSimilarity,
  n = 2,
): Array<[AngleName, number]> {
  return ANGLE_NAMES.map((k) => [k, position.jointDiffsDeg[k]] as [AngleName, number])
    .sort((a, b) => b[1] - a[1])
    .slice(0, n);
}
