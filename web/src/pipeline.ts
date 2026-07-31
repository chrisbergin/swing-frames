/**
 * End-to-end swing analysis: video file in, 8 labelled frames and their poses
 * out. The browser equivalent of process() in swing_frames.py.
 *
 * Pose detection is the entire cost here: decoding a frame takes a millisecond
 * or two, detecting a pose in one takes 40ms on a laptop and well over 100ms on
 * a phone. So the clip is walked in full, because frames can only be reached in
 * order, but pose runs on a sparse sample of it:
 *
 *   Pass 1  pose roughly 48 frames spread evenly across the clip, and find the
 *           swing's shape from those. Every other frame costs nothing.
 *   Pass 2  now that the positions are roughly known, keep the frames worth
 *           showing, and re-measure impact densely.
 *
 * The detection maths is unchanged and still matches the Python: it works on an
 * array of hand heights and derives its window sizes from the frame rate, so
 * feeding it 48 samples at the effective sample rate scales it automatically.
 *
 * Impact is the one position where a single frame matters, since it is a sharp
 * minimum a sparse sample steps straight over. That is what pass 2 re-measures.
 */

import { compareEventPoses, type Similarity } from "./core/angles";
import {
  EVENTS,
  LM,
  type EventFrames,
  type EventName,
  type Pose,
} from "./core/constants";
import {
  detectEvents,
  selectImpactFrame,
  SwingDetectionError,
  wristTrack,
} from "./core/events";
import {
  createLandmarker,
  firstPose,
  type ModelName,
} from "./pose/landmarker";
import {
  captureFrame,
  walkVideoFrames,
  type WalkResult,
} from "./video/decode";

export interface AnalyzeProgress {
  phase: "tracking" | "extracting";
  done: number;
  total: number;
  /** Below 1 when the decoder had to be slowed down to keep every frame. */
  playbackRate: number;
}

export interface AnalyzeOptions {
  model?: ModelName;
  /** Long-side cap for retained frames. Tiles are displayed small anyway. */
  maxTileSize?: number;
  /** How many poses to spread across the clip in the first pass. */
  coarseSamples?: number;
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
  /** How many poses the first pass actually ran. */
  posesRun: number;
  events: EventFrames;
  /** Pose at each position, in the coordinates of that position's tile. */
  eventPoses: Record<EventName, Pose | null>;
  tiles: Record<EventName, HTMLCanvasElement | null>;
  /** Speed the decoder settled on. Below 1 means this device needed slowing down. */
  playbackRate: number;
}

/**
 * Enough samples to resolve the swing's shape, few enough to stay quick.
 *
 * The structure being looked for is two hands-high episodes, and the episode
 * filter wants at least 3 samples inside one, so this needs to comfortably
 * exceed the number of "phases" in a swing. Around 48 leaves good margin.
 */
export const DEFAULT_COARSE_SAMPLES = 48;

/**
 * How many coarse samples either side of the estimated impact to re-measure
 * densely. Impact is a sharp minimum lasting a frame or two, so the coarse
 * pass only ever gets near it, never on it.
 */
const IMPACT_WINDOW_SAMPLES = 2;

/** Seconds between coarse samples. Zero disables sampling (pose every frame). */
export function sampleIntervalSeconds(
  durationSec: number,
  samples: number,
): number {
  if (!(durationSec > 0) || samples <= 0) return 0;
  return durationSec / samples;
}

/**
 * Above this share of lost frames the positions are not worth reporting.
 *
 * Dropped frames do not degrade gracefully. A run that lost 36 of 57 frames
 * still produced eight confident-looking tiles, but the frame rate came out at
 * 15fps instead of 27 and every position was wrong.
 */
export const MAX_DROP_RATE = 0.05;

/** Share of frames the decoder never handed over. */
export function dropRate(frameCount: number, droppedFrames: number): number {
  const total = frameCount + droppedFrames;
  return total === 0 ? 0 : droppedFrames / total;
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
 * Sparse sampling means most frames now cost nothing, so this should rarely be
 * needed; it remains the backstop for a device slow enough that even the
 * sampled frames overrun.
 */
export function retryPlaybackRate(
  msPerFrame: number,
  fps: number,
  currentRate: number,
): number {
  if (msPerFrame <= 0 || fps <= 0) return MIN_PLAYBACK_RATE;
  const frameIntervalMs = 1000 / fps;
  const rate = (frameIntervalMs * DUTY) / msPerFrame;
  return Math.max(MIN_PLAYBACK_RATE, Math.min(rate, currentRate * 0.8));
}

export async function analyzeSwing(
  file: Blob,
  {
    model = "full",
    maxTileSize = 720,
    coarseSamples = DEFAULT_COARSE_SAMPLES,
    playbackRate,
    onProgress,
    signal,
  }: AnalyzeOptions = {},
): Promise<SwingAnalysis> {
  // Pass 1: pose a sparse, evenly spaced sample of the clip.
  const coarsePoses: Array<Pose | null> = [];
  const sampleTimes: number[] = [];
  let walk!: WalkResult;
  let usedRate = playbackRate ?? 1;

  for (let attempt = 0; attempt < 2; attempt++) {
    // A fresh landmarker per attempt: VIDEO mode requires timestamps to keep
    // increasing, and a retry starts the clip over at zero.
    const landmarker = await createLandmarker("VIDEO", { model });
    coarsePoses.length = 0;
    sampleTimes.length = 0;
    let processingMs = 0;
    let interval = 0;
    let nextSampleAt = 0;
    const rate = usedRate;

    try {
      walk = await walkVideoFrames(
        file,
        (video, _index, mediaTimeSec) => {
          // Skipping is the whole point: this is the cheap path for most frames.
          if (mediaTimeSec + 1e-9 < nextSampleAt) return;
          nextSampleAt = mediaTimeSec + interval;

          const startedAt = performance.now();
          const result = landmarker.detectForVideo(video, mediaTimeSec * 1000);
          coarsePoses.push(
            firstPose(result, video.videoWidth, video.videoHeight),
          );
          sampleTimes.push(mediaTimeSec);
          processingMs += performance.now() - startedAt;

          onProgress?.({
            phase: "tracking",
            done: coarsePoses.length,
            total: coarseSamples,
            playbackRate: rate,
          });
        },
        {
          playbackRate: rate,
          onMetadata: ({ durationSec }) => {
            interval = sampleIntervalSeconds(durationSec, coarseSamples);
          },
          signal,
        },
      );
    } finally {
      landmarker.close();
    }

    if (dropRate(walk.frameCount, walk.droppedFrames) <= MAX_DROP_RATE) break;
    if (playbackRate !== undefined) break;

    usedRate = retryPlaybackRate(
      processingMs / Math.max(1, coarsePoses.length),
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

  if (sampleTimes.length < 10) {
    throw new SwingDetectionError(
      `Only ${sampleTimes.length} frames could be read from this clip. It may be ` +
        `too short, or in a format this browser cannot step through.`,
    );
  }

  // Detection runs on the sample, so the rate it sees is the sample rate.
  const span = sampleTimes[sampleTimes.length - 1] - sampleTimes[0];
  const effectiveFps = span > 0 ? (sampleTimes.length - 1) / span : 30;
  const { ys } = wristTrack(coarsePoses, effectiveFps);
  const coarse = detectEvents(ys, effectiveFps);

  // Positions come back as indices into the sample, so convert them to times,
  // which is the only thing that means anything across two different passes.
  const eventTimes = Object.fromEntries(
    EVENTS.map((name) => [name, sampleTimes[coarse[name]]]),
  ) as Record<EventName, number>;

  const sampleInterval = span / Math.max(1, sampleTimes.length - 1);

  // Pass 2: capture the frame nearest each position, and re-measure impact
  // densely, since it is a sharp minimum the coarse sample steps over.
  const imageLandmarker = await createLandmarker("IMAGE", { model });
  const best = new Map<
    EventName,
    { gap: number; frame: number; canvas: HTMLCanvasElement }
  >();
  // Impact's tile has to follow wherever the dense re-measure moves it, so the
  // whole window is retained rather than just the coarse guess. It is a few
  // frames either side, not the whole clip.
  const impactWindow: Array<{
    frame: number;
    y: number;
    canvas: HTMLCanvasElement;
  }> = [];

  try {
    await walkVideoFrames(
      file,
      (video, index, mediaTimeSec) => {
        // Wider than the other positions: the coarse guess for impact can sit
        // a full sample away from the real strike, and the window has to
        // bracket it or the dense re-measure has nothing to find.
        if (
          Math.abs(mediaTimeSec - eventTimes.impact) <=
          sampleInterval * IMPACT_WINDOW_SAMPLES
        ) {
          const pts = imageLandmarker.detect(video).landmarks[0];
          if (pts) {
            impactWindow.push({
              frame: index,
              canvas: captureFrame(video, maxTileSize),
              y:
                ((pts[LM.L_WRIST].y + pts[LM.R_WRIST].y) / 2) *
                video.videoHeight,
            });
          }
        }

        for (const name of EVENTS) {
          const gap = Math.abs(mediaTimeSec - eventTimes[name]);
          if (gap > sampleInterval) continue;
          const held = best.get(name);
          if (!held || gap < held.gap) {
            best.set(name, {
              gap,
              frame: index,
              canvas: captureFrame(video, maxTileSize),
            });
          }
        }

        onProgress?.({
          phase: "extracting",
          done: index + 1,
          total: walk.frameCount,
          playbackRate: usedRate,
        });
      },
      { playbackRate: usedRate, signal },
    );

    const tiles = Object.fromEntries(
      EVENTS.map((name) => [name, best.get(name)?.canvas ?? null]),
    ) as Record<EventName, HTMLCanvasElement | null>;

    // Impact moves to the densely measured frame, and its tile with it.
    if (impactWindow.length > 0) {
      const refined = selectImpactFrame(
        impactWindow.map(({ frame, y }) => ({ frame, y })),
        impactWindow[0].frame,
      );
      const at = impactWindow.find((m) => m.frame === refined);
      if (at) {
        tiles.impact = at.canvas;
        best.set("impact", { gap: 0, frame: at.frame, canvas: at.canvas });
      }
    }

    // Pose the 8 chosen frames off their tiles: eight detections rather than
    // one per frame, and angles are scale-invariant so the downscale is fine.
    const eventPoses = Object.fromEntries(
      EVENTS.map((name) => {
        const tile = tiles[name];
        if (!tile) return [name, null];
        return [name, firstPose(imageLandmarker.detect(tile), tile.width, tile.height)];
      }),
    ) as Record<EventName, Pose | null>;

    // Frame numbers of the frames actually shown, against the real clip.
    const events = Object.fromEntries(
      EVENTS.map((name) => [
        name,
        best.get(name)?.frame ?? Math.round(eventTimes[name] * walk.fps),
      ]),
    ) as EventFrames;

    return {
      fps: walk.fps,
      frameCount: walk.frameCount,
      droppedFrames: walk.droppedFrames,
      width: walk.width,
      height: walk.height,
      posesRun: coarsePoses.length,
      events,
      eventPoses,
      tiles,
      playbackRate: usedRate,
    };
  } finally {
    imageLandmarker.close();
  }
}

/** Joint-angle similarity between two analysed swings. */
export function compareAnalyses(
  a: SwingAnalysis,
  b: SwingAnalysis,
): Similarity {
  return compareEventPoses(a.eventPoses, b.eventPoses);
}
