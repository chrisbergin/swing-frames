/** Shared vocabulary for the swing pipeline: positions, landmarks, skeleton.
 *
 * Ported from swing_frames.py. Everything here is plain data so the core
 * logic stays testable in Node, with no browser or MediaPipe dependency.
 */

/** The 8 canonical swing positions, in swing order. */
export const EVENTS = [
  "address",
  "toe_up",
  "mid_backswing",
  "top",
  "mid_downswing",
  "impact",
  "mid_follow_through",
  "finish",
] as const;

export type EventName = (typeof EVENTS)[number];

/** Frame index chosen for each position. */
export type EventFrames = Record<EventName, number>;

/** A landmark in pixel coordinates. y grows downward, so high hands = small y. */
export interface Point {
  x: number;
  y: number;
}

/** One frame's pose: the 33 MediaPipe landmarks, or null if none was detected. */
export type Pose = Point[];

/** Indices into the 33-point MediaPipe Pose model (body only, face skipped). */
export const LM = {
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_ELBOW: 13,
  R_ELBOW: 14,
  L_WRIST: 15,
  R_WRIST: 16,
  L_HIP: 23,
  R_HIP: 24,
  L_KNEE: 25,
  R_KNEE: 26,
  L_ANKLE: 27,
  R_ANKLE: 28,
} as const;

/** Body-only skeleton for the overlay (face landmarks are not drawn). */
export const POSE_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28], [27, 29], [29, 31], [28, 30], [30, 32],
  [27, 31], [28, 32],
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22],
];

/**
 * Round to one decimal, matching the precision the Python script writes to
 * events.json and similarity_*.json so the two outputs can be diffed.
 *
 * Python's round() breaks exact .5 ties to even and this breaks them upward,
 * which can differ in the last decimal place. Angle values are continuous
 * floats, so a tie is possible but not worth carrying banker's rounding for.
 */
export function round1(x: number): number {
  return Math.round(x * 10) / 10;
}
