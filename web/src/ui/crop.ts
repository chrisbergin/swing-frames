/**
 * Pose-anchored display crop, so two golfers filmed at different distances
 * show at the same size with their feet on the same line.
 *
 * The anchor is computed once per video from the median across all 8 event
 * poses, not per position: one blurry frame's bad pose would otherwise zoom
 * or shift its crop, and the median keeps the background still while stepping
 * through positions. It reads the shoulder-to-ankle span rather than the full
 * landmark extent, because hands travel far above the shoulders mid-swing.
 *
 * The window shape is fitted jointly across the videos being shown, so the
 * narrower clip does not need letterboxing to keep the shapes equal.
 */

import { LM, type Pose } from "../core/constants";

/** Source-pixel rectangle to display. */
export interface CropRect {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

/** Where a video's golfer stands: body scale, feet line, horizontal centre. */
export interface Anchor {
  span: number;
  feetY: number;
  cx: number;
}

/** Crop height in units of the shoulder-to-ankle span: head above, club below. */
const CROP_HEIGHT_SPANS = 2.0;
/** How far down the crop the feet sit. */
const FEET_LINE = 0.84;
/** Soles sit a touch below the ankle landmarks. */
const SOLE_DROP = 0.06;
/** Widest window shape (width over height); portrait suits a standing golfer. */
const MAX_ASPECT = 0.75;
/** Never squeeze narrower than this, however narrow a clip is. */
const MIN_ASPECT = 0.5;

/**
 * One pose's anchor, or null when the pose cannot be a standing golfer:
 * too small in frame, or shoulders/hips/ankles not in body order, which is
 * what a garbage detection on a blurry frame looks like.
 */
export function poseAnchor(pose: Pose, frameHeight: number): Anchor | null {
  const shoulderTop = Math.min(pose[LM.L_SHOULDER].y, pose[LM.R_SHOULDER].y);
  const ankleBottom = Math.max(pose[LM.L_ANKLE].y, pose[LM.R_ANKLE].y);
  const hipY = (pose[LM.L_HIP].y + pose[LM.R_HIP].y) / 2;
  const span = ankleBottom - shoulderTop;
  if (!(span > 0) || span < frameHeight * 0.05) return null;
  if (!(shoulderTop < hipY && hipY < ankleBottom)) return null;
  return {
    span,
    feetY: ankleBottom + SOLE_DROP * span,
    cx: (pose[LM.L_HIP].x + pose[LM.R_HIP].x) / 2,
  };
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) / 2 : s[mid];
}

/** The video's anchor: medians over every usable event pose. */
export function videoAnchor(
  poses: ReadonlyArray<Pose | null>,
  frameHeight: number,
): Anchor | null {
  const anchors: Anchor[] = [];
  for (const pose of poses) {
    if (!pose) continue;
    const anchor = poseAnchor(pose, frameHeight);
    if (anchor) anchors.push(anchor);
  }
  if (anchors.length === 0) return null;
  return {
    span: median(anchors.map((a) => a.span)),
    feetY: median(anchors.map((a) => a.feetY)),
    cx: median(anchors.map((a) => a.cx)),
  };
}

/**
 * Crops for the videos shown together, sharing one window shape.
 *
 * The shape narrows (down to MIN_ASPECT) until it fits inside every clip, so
 * a zoomed-in reference does not get a black bar pushing its golfer sideways.
 * Horizontal position then clamps inside the frame: sideways placement
 * carries no alignment meaning, unlike the feet line, which never shifts.
 */
export function resolveCrops(
  sides: ReadonlyArray<{ anchor: Anchor | null; frameWidth: number }>,
): Array<CropRect | null> {
  let aspect = MAX_ASPECT;
  for (const { anchor, frameWidth } of sides) {
    if (anchor) {
      aspect = Math.min(aspect, frameWidth / (CROP_HEIGHT_SPANS * anchor.span));
    }
  }
  aspect = Math.max(aspect, MIN_ASPECT);

  return sides.map(({ anchor, frameWidth }) => {
    if (!anchor) return null;
    const sh = CROP_HEIGHT_SPANS * anchor.span;
    const sy = anchor.feetY - FEET_LINE * sh;
    const sw = sh * aspect;
    let sx = anchor.cx - sw / 2;
    sx =
      sw <= frameWidth
        ? Math.min(Math.max(sx, 0), frameWidth - sw)
        : (frameWidth - sw) / 2;
    return { sx, sy, sw, sh };
  });
}
