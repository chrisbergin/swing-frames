/**
 * End-to-end swing analysis: video file in, 8 labelled frames and a set of
 * poses out. The browser equivalent of process() in swing_frames.py.
 *
 * Runs as two sequential decode passes rather than one, because holding a
 * whole clip's frames in memory is not an option on a phone: a 240fps slo-mo
 * swing is hundreds of frames, and at full resolution that is gigabytes.
 *
 *   Pass 1  pose-detect every frame in VIDEO mode, keep only the landmarks,
 *           throw the pixels away. Detection needs nothing else.
 *   Pass 2  now that the positions are known, keep just the frames that will
 *           be shown, and re-measure the impact window in IMAGE mode.
 *
 * Decoding twice is cheap next to pose detection, which dominates either way.
 */

import { compareSwings, type Similarity } from "./core/angles";
import {
  EVENTS,
  LM,
  type EventFrames,
  type EventName,
  type Pose,
} from "./core/constants";
import {
  detectEvents,
  impactRefineWindow,
  selectImpactFrame,
  SwingDetectionError,
  wristTrack,
} from "./core/events";
import { createLandmarker, firstPose, type ModelName } from "./pose/landmarker";
import { captureFrame, walkVideoFrames } from "./video/decode";

export interface AnalyzeProgress {
  phase: "tracking" | "extracting";
  done: number;
  /** Zero until pass 1 has finished, since the frame count is not known up front. */
  total: number;
}

export interface AnalyzeOptions {
  model?: ModelName;
  /** Long-side cap for retained frames. Tiles are displayed small anyway. */
  maxTileSize?: number;
  playbackRate?: number;
  onProgress?: (progress: AnalyzeProgress) => void;
  signal?: AbortSignal;
}

export interface SwingAnalysis {
  fps: number;
  frameCount: number;
  droppedFrames: number;
  width: number;
  height: number;
  events: EventFrames;
  /** Per-frame poses from pass 1, in source pixel coordinates. */
  poses: Array<Pose | null>;
  /** The 8 displayed frames, already scaled down. */
  tiles: Record<EventName, HTMLCanvasElement | null>;
  /** Multiply a source-pixel pose coordinate by this to land on a tile. */
  tileScale: number;
}

/**
 * Above this share of lost frames the positions are not worth reporting.
 *
 * Dropped frames do not degrade gracefully. A run that lost 36 of 57 frames
 * still produced eight confident-looking tiles, but the frame rate came out at
 * 15fps instead of 27 and every position was wrong. Failing loudly beats
 * handing back a plausible answer that happens to be nonsense.
 */
export const MAX_DROP_RATE = 0.05;

/** Share of frames the decoder never handed over. */
export function dropRate(frameCount: number, droppedFrames: number): number {
  const total = frameCount + droppedFrames;
  return total === 0 ? 0 : droppedFrames / total;
}

/**
 * Which frames pass 2 needs to hold on to: the 8 positions, plus the whole
 * impact re-measure window, because refinement can move impact to any frame
 * in it and that frame still has to be displayable.
 */
export function framesToRetain(
  events: EventFrames,
  windowLo: number,
  windowHi: number,
): Set<number> {
  const keep = new Set<number>(EVENTS.map((e) => events[e]));
  for (let i = windowLo; i < windowHi; i++) keep.add(i);
  return keep;
}

export async function analyzeSwing(
  file: Blob,
  {
    model = "full",
    maxTileSize = 720,
    playbackRate = 1,
    onProgress,
    signal,
  }: AnalyzeOptions = {},
): Promise<SwingAnalysis> {
  // Pass 1: landmarks only.
  const videoLandmarker = await createLandmarker("VIDEO", { model });
  const poses: Array<Pose | null> = [];
  let walk;
  try {
    walk = await walkVideoFrames(
      file,
      (video, index, mediaTimeSec) => {
        const result = videoLandmarker.detectForVideo(video, mediaTimeSec * 1000);
        poses.push(firstPose(result, video.videoWidth, video.videoHeight));
        onProgress?.({ phase: "tracking", done: index + 1, total: 0 });
      },
      { playbackRate, signal },
    );
  } finally {
    videoLandmarker.close();
  }

  if (dropRate(walk.frameCount, walk.droppedFrames) > MAX_DROP_RATE) {
    const total = walk.frameCount + walk.droppedFrames;
    throw new SwingDetectionError(
      `Could not keep up with the video: ${walk.droppedFrames} of ${total} ` +
        `frames were missed, which makes the detected positions unreliable. ` +
        `Try the lite model, or a shorter clip.`,
    );
  }

  const { ys } = wristTrack(poses, walk.fps);
  const tracked = detectEvents(ys, walk.fps);
  const { lo, hi } = impactRefineWindow(tracked, walk.fps, poses.length);
  const retain = framesToRetain(tracked, lo, hi);

  // Pass 2: keep the frames worth showing, and re-measure impact without the
  // temporal prior that makes VIDEO mode lag the hands through the strike.
  const imageLandmarker = await createLandmarker("IMAGE", { model });
  const kept = new Map<number, HTMLCanvasElement>();
  const measured: Array<{ frame: number; y: number }> = [];
  try {
    await walkVideoFrames(
      file,
      (video, index) => {
        if (index >= lo && index < hi) {
          const pts = imageLandmarker.detect(video).landmarks[0];
          if (pts) {
            measured.push({
              frame: index,
              y:
                ((pts[LM.L_WRIST].y + pts[LM.R_WRIST].y) / 2) *
                video.videoHeight,
            });
          }
        }
        if (retain.has(index)) kept.set(index, captureFrame(video, maxTileSize));
        onProgress?.({
          phase: "extracting",
          done: index + 1,
          total: walk.frameCount,
        });
      },
      { playbackRate, signal },
    );
  } finally {
    imageLandmarker.close();
  }

  const events: EventFrames = {
    ...tracked,
    impact: selectImpactFrame(measured, tracked.impact),
  };

  const tiles = Object.fromEntries(
    EVENTS.map((name) => [name, kept.get(events[name]) ?? null]),
  ) as Record<EventName, HTMLCanvasElement | null>;

  const anyTile = EVENTS.map((e) => tiles[e]).find((t) => t !== null);

  return {
    fps: walk.fps,
    frameCount: walk.frameCount,
    droppedFrames: walk.droppedFrames,
    width: walk.width,
    height: walk.height,
    events,
    poses,
    tiles,
    tileScale: anyTile && walk.width ? anyTile.width / walk.width : 1,
  };
}

/** Joint-angle similarity between two analysed swings. */
export function compareAnalyses(
  a: SwingAnalysis,
  b: SwingAnalysis,
): Similarity {
  return compareSwings(a.poses, a.events, b.poses, b.events);
}
