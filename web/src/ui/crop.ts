/**
 * Pose-anchored display crop, so two golfers filmed at different distances
 * show at the same size with their feet on the same line.
 *
 * Anchored on the shoulder-to-ankle span rather than the full landmark extent:
 * hands travel far above the shoulders mid-swing, and a crop that chased them
 * would change the golfer's size from one position to the next.
 */

import { LM, type Pose } from "../core/constants";

/** Source-pixel rectangle to display. May extend past the frame: the overhang
 * is letterboxed rather than shifted, so alignment never bends near an edge. */
export interface CropRect {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

/** Crop height in units of the shoulder-to-ankle span: head above, club below. */
const CROP_HEIGHT_SPANS = 2.0;
/** How far down the crop the feet sit. */
const FEET_LINE = 0.84;
/** Soles sit a touch below the ankle landmarks. */
const SOLE_DROP = 0.06;
/** Width over height of the crop window; portrait suits a standing golfer. */
const ASPECT = 0.75;

/**
 * The crop that puts this pose's golfer at a standard size and position, or
 * null when the pose is too degenerate to anchor one (which also covers a
 * pose that was detected on something that is not a standing human).
 */
export function golferCrop(pose: Pose, frameHeight: number): CropRect | null {
  const shoulderTop = Math.min(pose[LM.L_SHOULDER].y, pose[LM.R_SHOULDER].y);
  const ankleBottom = Math.max(pose[LM.L_ANKLE].y, pose[LM.R_ANKLE].y);
  const span = ankleBottom - shoulderTop;
  if (!(span > 0) || span < frameHeight * 0.05) return null;

  const feetY = ankleBottom + SOLE_DROP * span;
  const sh = CROP_HEIGHT_SPANS * span;
  const sw = sh * ASPECT;
  const sy = feetY - FEET_LINE * sh;
  const sx = (pose[LM.L_HIP].x + pose[LM.R_HIP].x) / 2 - sw / 2;
  return { sx, sy, sw, sh };
}
