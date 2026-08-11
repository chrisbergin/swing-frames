/**
 * Drawing one extracted frame: the image, its skeleton, and the optional
 * align crop.
 *
 * Lives on its own because two callers need identical output. The tile on
 * screen is one; the saved comparison sheet is the other, and a sheet that
 * drew its frames even slightly differently from the app would be a confusing
 * thing to hand someone.
 */

import type { Pose } from "../core/constants";
import type { CropRect } from "./crop";
import { drawPose } from "./draw";

/** Rendered height of an aligned (cropped) frame, in canvas pixels. */
export const ALIGNED_TILE_HEIGHT = 720;

/** What a position currently shows: the detected frame or a nudged one. */
export interface FrameView {
  tile: HTMLCanvasElement | null;
  pose: Pose | null;
  crop?: CropRect | null;
}

/**
 * Size `canvas` to the frame and draw it, cropped and skeletoned.
 *
 * Returns false when there is nothing to draw, so callers can show their own
 * placeholder rather than an empty canvas.
 */
export function drawFrame(canvas: HTMLCanvasElement, view: FrameView): boolean {
  const { tile, pose, crop } = view;
  if (!tile) return false;
  const ctx = canvas.getContext("2d");
  if (!ctx) return false;

  if (!crop) {
    canvas.width = tile.width;
    canvas.height = tile.height;
    ctx.drawImage(tile, 0, 0);
    // Poses are measured off the tile, so they are already in its coordinates.
    if (pose) drawPose(ctx, pose, 1);
    return true;
  }

  const scale = ALIGNED_TILE_HEIGHT / crop.sh;
  canvas.width = Math.round(crop.sw * scale);
  canvas.height = ALIGNED_TILE_HEIGHT;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // The crop may reach past the frame; draw the part that exists and leave the
  // overhang black, so the alignment never bends near an edge.
  const ox = Math.max(crop.sx, 0);
  const oy = Math.max(crop.sy, 0);
  const ox2 = Math.min(crop.sx + crop.sw, tile.width);
  const oy2 = Math.min(crop.sy + crop.sh, tile.height);
  if (ox2 > ox && oy2 > oy) {
    ctx.drawImage(
      tile,
      ox,
      oy,
      ox2 - ox,
      oy2 - oy,
      (ox - crop.sx) * scale,
      (oy - crop.sy) * scale,
      (ox2 - ox) * scale,
      (oy2 - oy) * scale,
    );
  }
  if (pose) {
    ctx.save();
    ctx.setTransform(scale, 0, 0, scale, -crop.sx * scale, -crop.sy * scale);
    drawPose(ctx, pose, 1);
    ctx.restore();
  }
  return true;
}

/** The same drawing, into a canvas of its own. */
export function renderFrame(view: FrameView): HTMLCanvasElement | null {
  const canvas = document.createElement("canvas");
  return drawFrame(canvas, view) ? canvas : null;
}
