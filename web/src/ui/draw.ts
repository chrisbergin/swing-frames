/**
 * Pose skeleton overlay. Port of draw_pose() in swing_frames.py.
 *
 * The Python's colours are BGR, as OpenCV wants; the equivalent RGB is used
 * here so the overlay looks the same as the frames the prototype produces.
 */

import { POSE_CONNECTIONS, type Pose } from "../core/constants";

const LIMB_COLOR = "rgb(140, 255, 0)";
const JOINT_COLOR = "rgb(255, 140, 0)";

/** Face landmarks are not drawn: the swing is legible from the body alone. */
const FIRST_BODY_LANDMARK = 11;

export function drawPose(
  ctx: CanvasRenderingContext2D,
  pose: Pose,
  scale: number,
): void {
  const x = (i: number) => pose[i].x * scale;
  const y = (i: number) => pose[i].y * scale;

  // Keep the skeleton readable whatever size the tile ended up.
  const unit = Math.max(1, ctx.canvas.width / 360);

  ctx.save();
  ctx.lineWidth = 2 * unit;
  ctx.strokeStyle = LIMB_COLOR;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [a, b] of POSE_CONNECTIONS) {
    ctx.moveTo(x(a), y(a));
    ctx.lineTo(x(b), y(b));
  }
  ctx.stroke();

  ctx.fillStyle = JOINT_COLOR;
  for (let i = FIRST_BODY_LANDMARK; i < pose.length; i++) {
    ctx.beginPath();
    ctx.arc(x(i), y(i), 4 * unit, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
