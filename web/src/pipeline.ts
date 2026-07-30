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
import {
  captureFrame,
  walkVideoFrames,
  type WalkResult,
} from "./video/decode";

export interface AnalyzeProgress {
  phase: "tracking" | "extracting";
  done: number;
  /** Zero until pass 1 has finished, since the frame count is not known up front. */
  total: number;
  /** Below 1 when the decoder had to be slowed down to keep every frame. */
  playbackRate: number;
}

export interface AnalyzeOptions {
  model?: ModelName;
  /** Long-side cap for retained frames. Tiles are displayed small anyway. */
  maxTileSize?: number;
  /** Pin the playback speed. Left unset, it steps down until no frames drop. */
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
  /** Speed the decoder settled on. Below 1 means this device needed slowing down. */
  playbackRate: number;
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

/**
 * Leave this much headroom when working out how far to slow playback down.
 * Per-frame cost varies, so aiming to fill the whole interval would drop the
 * slow frames.
 */
const DUTY = 0.7;

/** Never slow below this: past here the wait is worse than the failure. */
const MIN_PLAYBACK_RATE = 0.05;

/**
 * Playback speed at which a frame's processing fits inside a frame interval.
 *
 * Pausing the media clock during the frame callback is supposed to make
 * processing speed irrelevant, and on desktop Chrome it does. Phones are
 * slower and the pause does not always take hold before the next frame is
 * presented, so the remaining lever is wall-clock time between frames: at
 * quarter speed a 30fps clip presents a frame every 133ms instead of every
 * 33ms. Deriving the rate from measured cost beats guessing at a ladder of
 * speeds, because it converges in one retry instead of several.
 */
export function retryPlaybackRate(
  msPerFrame: number,
  fps: number,
  currentRate: number,
): number {
  if (msPerFrame <= 0 || fps <= 0) return MIN_PLAYBACK_RATE;
  const frameIntervalMs = 1000 / fps;
  const rate = (frameIntervalMs * DUTY) / msPerFrame;
  // Only ever slower than the attempt that just failed.
  return Math.max(MIN_PLAYBACK_RATE, Math.min(rate, currentRate * 0.8));
}

export async function analyzeSwing(
  file: Blob,
  {
    model = "full",
    maxTileSize = 720,
    playbackRate,
    onProgress,
    signal,
  }: AnalyzeOptions = {},
): Promise<SwingAnalysis> {
  // Pass 1: landmarks only. Retried slower if the decoder could not keep up,
  // since dropped frames do not degrade gracefully.
  const poses: Array<Pose | null> = [];
  let walk!: WalkResult;
  let usedRate = playbackRate ?? 1;

  // Two attempts at most: full speed, then one at a rate derived from how long
  // this device actually took per frame.
  for (let attempt = 0; attempt < 2; attempt++) {
    // A fresh landmarker per attempt: VIDEO mode requires timestamps to keep
    // increasing, and a retry starts the clip over at zero.
    const videoLandmarker = await createLandmarker("VIDEO", { model });
    poses.length = 0;
    let processingMs = 0;
    const rate = usedRate;

    try {
      walk = await walkVideoFrames(
        file,
        (video, index, mediaTimeSec) => {
          const startedAt = performance.now();
          const result = videoLandmarker.detectForVideo(
            video,
            mediaTimeSec * 1000,
          );
          poses.push(firstPose(result, video.videoWidth, video.videoHeight));
          processingMs += performance.now() - startedAt;
          onProgress?.({
            phase: "tracking",
            done: index + 1,
            total: 0,
            playbackRate: rate,
          });
        },
        { playbackRate: rate, signal },
      );
    } finally {
      videoLandmarker.close();
    }

    if (dropRate(walk.frameCount, walk.droppedFrames) <= MAX_DROP_RATE) break;
    // A pinned rate is the caller's decision, so do not second-guess it.
    if (playbackRate !== undefined) break;

    usedRate = retryPlaybackRate(
      processingMs / Math.max(1, walk.frameCount),
      walk.fps,
      rate,
    );
  }

  if (dropRate(walk.frameCount, walk.droppedFrames) > MAX_DROP_RATE) {
    const total = walk.frameCount + walk.droppedFrames;
    throw new SwingDetectionError(
      `Could not keep up with the video, even slowed down: ${walk.droppedFrames} ` +
        `of ${total} frames were missed at ${walk.fps.toFixed(1)}fps, which makes ` +
        `the detected positions unreliable. Try the lite model, or a shorter clip.`,
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
          playbackRate: usedRate,
        });
      },
      // Whatever speed pass 1 needed, since a device that could not keep up
      // with pose detection will not keep up with capturing frames either.
      { playbackRate: usedRate, signal },
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
    playbackRate: usedRate,
  };
}

/** Joint-angle similarity between two analysed swings. */
export function compareAnalyses(
  a: SwingAnalysis,
  b: SwingAnalysis,
): Similarity {
  return compareSwings(a.poses, a.events, b.poses, b.events);
}
