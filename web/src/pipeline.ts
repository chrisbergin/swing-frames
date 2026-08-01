/**
 * End-to-end swing analysis: video file in, 8 labelled frames and their poses
 * out. The browser equivalent of process() in swing_frames.py.
 *
 * The clip is never played. Frames are fetched by seeking directly to the times
 * of interest, which is what keeps this usable on a phone:
 *
 *   - requestVideoFrameCallback only fires during playback, so walking a clip
 *     costs its full running time no matter how few frames are wanted.
 *   - Worse, keeping every frame meant pausing and resuming the media clock
 *     around each one, and on a phone that overhead lands on every frame
 *     whether or not it is used. 324 frames took minutes.
 *
 * The standing "never seek" rule still holds wherever a frame *number* matters,
 * and it is respected: every seek reports the time it actually landed on, and
 * nothing here depends on hitting an exact frame.
 *
 * Shape of the work, which is Chris's coarse-to-fine idea:
 *
 *   1. Pose ~48 points spread evenly across the clip and read the swing's
 *      structure from those.
 *   2. Narrow impact twice, since it is a sharp minimum a coarse sample steps
 *      straight over, each round searching a window a fraction of the last.
 *   3. Fetch the 8 chosen frames for display and scoring.
 */

import { compareEventPoses, type Similarity } from "./core/angles";
import {
  EVENTS,
  LM,
  type EventName,
  type Pose,
} from "./core/constants";
import { detectEvents, SwingDetectionError, wristTrack } from "./core/events";
import { createLandmarker, firstPose, type ModelName } from "./pose/landmarker";
import {
  drawRotatedFrame,
  grabFrameAt,
  loadVideo,
  sampleAtTimes,
  type Rotation,
} from "./video/decode";
import { WebCodecsClip, webCodecsSupported } from "./video/webcodecs";

export interface AnalyzeProgress {
  phase: "tracking" | "refining" | "extracting";
  done: number;
  total: number;
}

export interface AnalyzeOptions {
  model?: ModelName;
  /** Clockwise rotation for clips without rotation metadata. */
  rotate?: Rotation;
  /** Long-side cap for retained frames. Tiles are displayed small anyway. */
  maxTileSize?: number;
  /** Fixed pose count for the first pass. Defaults to scaling with duration. */
  coarseSamples?: number;
  onProgress?: (progress: AnalyzeProgress) => void;
  signal?: AbortSignal;
}

export interface SwingAnalysis {
  durationSec: number;
  width: number;
  height: number;
  /** How many pose detections the whole analysis ran. */
  posesRun: number;
  /** How many of those found a person. Low against posesRun means the golfer
   * was not visible: cropped, tiny, sideways, or absent. */
  posesFound: number;
  /** How the displayed frames were captured. "video" means WebCodecs was
   * unavailable or could not read the clip and it fell back to the <video>
   * element. */
  captureMethod: "webcodecs" | "video";
  /**
   * When each position happens, in seconds.
   *
   * Time, not frame number. Nothing here ever counts frames: seeking reports
   * the time it landed on, and a seek's currentTime is the time requested
   * rather than a frame boundary, so frame numbers cannot be inferred from it
   * and are not worth inventing.
   */
  eventTimes: Record<EventName, number>;
  /** Pose at each position, in the coordinates of that position's tile. */
  eventPoses: Record<EventName, Pose | null>;
  tiles: Record<EventName, HTMLCanvasElement | null>;
}

/**
 * Enough samples to resolve the swing's shape, few enough to stay quick.
 *
 * Detection looks for two hands-high episodes and wants at least 3 samples
 * inside one, so this has to comfortably exceed the number of phases in a
 * swing. Around 48 leaves good margin.
 */
export const DEFAULT_COARSE_SAMPLES = 48;

/**
 * How finely to sample, and the bounds on it.
 *
 * A fixed sample count gives a short clip fine resolution and a long one
 * almost none: 48 samples is one every 0.09s across a 4 second phone video,
 * but one every 0.63s across a 30 second analysis video, which is coarse
 * enough to land in the wrong freeze frame entirely. Measured on the pro
 * reference clip, that cost about 15 points of similarity score. So aim for a
 * time resolution instead, with a ceiling to keep a long clip from costing
 * hundreds of detections.
 */
const TARGET_SAMPLE_INTERVAL_SEC = 0.1;
const MAX_COARSE_SAMPLES = 150;

/** Sample count giving roughly a fixed time resolution, within bounds. */
export function coarseSampleCount(durationSec: number): number {
  if (!(durationSec > 0)) return DEFAULT_COARSE_SAMPLES;
  const wanted = Math.round(durationSec / TARGET_SAMPLE_INTERVAL_SEC);
  return Math.max(
    DEFAULT_COARSE_SAMPLES,
    Math.min(wanted, MAX_COARSE_SAMPLES),
  );
}

/** Samples per impact-narrowing round, and how many rounds to run (the
 * <video> fallback path only; WebCodecs poses the whole window at once). */
const REFINE_SAMPLES = 12;
const REFINE_ROUNDS = 2;

/**
 * Half-width of the window re-posed around the coarse impact on WebCodecs.
 *
 * Generous on purpose: the coarse impact comes from the device-dependent
 * <video> pass and can be off, so the window must still contain the true
 * strike for the deterministic re-measure to land on it.
 */
const IMPACT_WINDOW_SEC = 0.4;

/** Long side for the refine poses; smaller is faster and angles are scale-free. */
const REFINE_POSE_SIZE = 480;

/**
 * The impact time from re-measured wrist heights.
 *
 * Hands lowest is the largest y; ball contact comes fractionally after that and
 * neighbouring frames measure within noise of each other, so take the last (in
 * time) within a hair of the bottom. Returns null if nothing was measured.
 */
function pickImpactTime(
  measured: ReadonlyArray<{ time: number; y: number }>,
): number | null {
  if (measured.length === 0) return null;
  let bottom = -Infinity;
  let peak = Infinity;
  for (const m of measured) {
    if (m.y > bottom) bottom = m.y;
    if (m.y < peak) peak = m.y;
  }
  const eps = 0.05 * (bottom - peak);
  const candidates = measured
    .filter((m) => m.y >= bottom - eps)
    .sort((a, b) => a.time - b.time);
  return candidates[candidates.length - 1].time;
}


/** Seconds between coarse samples. */
export function sampleIntervalSeconds(
  durationSec: number,
  samples: number,
): number {
  if (!(durationSec > 0) || samples <= 0) return 0;
  return durationSec / samples;
}

/** Evenly spaced sample times, offset half a step so none sits on an edge. */
export function planSampleTimes(
  durationSec: number,
  samples: number,
): number[] {
  const interval = sampleIntervalSeconds(durationSec, samples);
  if (interval === 0) return [0];
  return Array.from({ length: samples }, (_, i) => (i + 0.5) * interval);
}

/** `count` times spanning `centre` +/- `halfWidth`, clamped to the clip. */
export function planWindowTimes(
  centre: number,
  halfWidth: number,
  count: number,
  durationSec: number,
): number[] {
  if (count < 2) return [centre];
  const step = (2 * halfWidth) / (count - 1);
  return Array.from({ length: count }, (_, i) =>
    Math.max(0, Math.min(centre - halfWidth + i * step, durationSec)),
  );
}

/** Wrist-midpoint height in pixels, or NaN if no pose was found. */
function wristHeight(pose: Pose | null): number {
  if (!pose) return NaN;
  return (pose[LM.L_WRIST].y + pose[LM.R_WRIST].y) / 2;
}

export async function analyzeSwing(
  file: Blob,
  {
    model = "full",
    rotate = 0,
    maxTileSize = 720,
    coarseSamples,
    onProgress,
    signal,
  }: AnalyzeOptions = {},
): Promise<SwingAnalysis> {
  const landmarker = await createLandmarker("IMAGE", { model });
  let posesRun = 0;
  let posesFound = 0;
  // Filled in as soon as they are known, so a failure can still report them.
  let clip = { durationSec: 0, width: 0, height: 0 };

  // With a rotation set, detection runs on this scratch canvas instead of the
  // video element. Reuse is safe: MediaPipe reads it synchronously.
  const scratch = rotate === 0 ? null : document.createElement("canvas");
  const detectCurrentFrame = (video: HTMLVideoElement): Pose | null => {
    let pose: Pose | null;
    if (scratch) {
      drawRotatedFrame(scratch, video, rotate);
      pose = firstPose(landmarker.detect(scratch), scratch.width, scratch.height);
    } else {
      pose = firstPose(
        landmarker.detect(video),
        video.videoWidth,
        video.videoHeight,
      );
    }
    posesRun++;
    if (pose) posesFound++;
    return pose;
  };

  try {
    // 1. Coarse pass: the shape of the swing.
    const coarsePoses: Array<Pose | null> = [];
    const coarseTimes: number[] = [];
    let samples = coarseSamples ?? DEFAULT_COARSE_SAMPLES;

    const meta = await sampleAtTimes(
      file,
      (durationSec) => {
        clip.durationSec = durationSec;
        samples = coarseSamples ?? coarseSampleCount(durationSec);
        return planSampleTimes(durationSec, samples);
      },
      ({ video, timeSec }) => {
        clip.width = video.videoWidth;
        clip.height = video.videoHeight;
        coarsePoses.push(detectCurrentFrame(video));
        coarseTimes.push(timeSec);
        onProgress?.({
          phase: "tracking",
          done: coarsePoses.length,
          total: samples,
        });
      },
      { signal },
    );

    if (coarseTimes.length < 10) {
      throw new SwingDetectionError(
        `Only ${coarseTimes.length} frames could be read from this clip. It may ` +
          `be too short, or in a format this browser cannot read.`,
      );
    }

    const span = coarseTimes[coarseTimes.length - 1] - coarseTimes[0];
    const effectiveFps = span > 0 ? (coarseTimes.length - 1) / span : 30;
    const { ys } = wristTrack(coarsePoses, effectiveFps);
    const coarse = detectEvents(ys, effectiveFps);

    const eventTimes = Object.fromEntries(
      EVENTS.map((name) => [name, coarseTimes[coarse[name]]]),
    ) as Record<EventName, number>;

    // Open a WebCodecs clip once, shared by the impact refinement and the
    // display grab. Decoding never touches the <video> surface, so both come
    // out identical on desktop and phone; failure (unsupported browser, a
    // container the demuxer cannot read) leaves it null and everything falls
    // back to the device-dependent <video> path, no worse off than before.
    let wcClip: WebCodecsClip | null = null;
    if (webCodecsSupported()) {
      try {
        wcClip = await WebCodecsClip.open(file);
      } catch {
        wcClip = null;
      }
    }
    const captureMethod: "webcodecs" | "video" = wcClip ? "webcodecs" : "video";

    // 2. Narrow impact.
    if (wcClip) {
      // Re-measure a generous window around the coarse impact on decoded
      // frames. Wide enough to hold the true strike even when the coarse
      // estimate (from the device-dependent <video> pass) is off, so the
      // result converges to the same frame on every device.
      const lo = Math.max(eventTimes.top, eventTimes.impact - IMPACT_WINDOW_SEC);
      const hi = Math.min(eventTimes.finish, eventTimes.impact + IMPACT_WINDOW_SEC);
      const measured: Array<{ time: number; y: number }> = [];
      const estTotal = Math.max(1, Math.round((hi - lo) * 60));
      await wcClip.forEachFrameInWindow(
        lo,
        hi,
        REFINE_POSE_SIZE,
        (canvas, t) => {
          const y = wristHeight(firstPose(landmarker.detect(canvas), canvas.width, canvas.height));
          posesRun++;
          if (!Number.isNaN(y)) {
            posesFound++;
            measured.push({ time: t, y });
          }
          onProgress?.({ phase: "refining", done: measured.length, total: estTotal });
        },
        rotate,
        signal,
      );
      const picked = pickImpactTime(measured);
      if (picked !== null) eventTimes.impact = picked;
    } else {
      // Fallback: two-round narrowing over the <video> element, a window a
      // fraction of the last each round.
      let halfWidth = sampleIntervalSeconds(meta.durationSec, samples) * 1.5;
      for (let round = 0; round < REFINE_ROUNDS; round++) {
        const measured: Array<{ time: number; y: number }> = [];
        await sampleAtTimes(
          file,
          (durationSec) =>
            planWindowTimes(eventTimes.impact, halfWidth, REFINE_SAMPLES, durationSec),
          ({ video, timeSec }) => {
            const y = wristHeight(detectCurrentFrame(video));
            if (!Number.isNaN(y)) measured.push({ time: timeSec, y });
            onProgress?.({
              phase: "refining",
              done: round * REFINE_SAMPLES + measured.length,
              total: REFINE_ROUNDS * REFINE_SAMPLES,
            });
          },
          { signal },
        );
        const picked = pickImpactTime(measured);
        if (picked === null) break;
        eventTimes.impact = picked;
        halfWidth = (2 * halfWidth) / (REFINE_SAMPLES - 1);
      }
    }

    // 3. Fetch the frames that will actually be shown.
    const tiles = {} as Record<EventName, HTMLCanvasElement | null>;
    const eventPoses = {} as Record<EventName, Pose | null>;
    for (const name of EVENTS) {
      tiles[name] = null;
      eventPoses[name] = null;
    }

    const order = [...EVENTS].sort((a, b) => eventTimes[a] - eventTimes[b]);
    const targetTimes = order.map((name) => eventTimes[name]);

    let grabbed: Array<{ canvas: HTMLCanvasElement; timeSec: number }> | null = null;
    if (wcClip) {
      grabbed = await wcClip.grab(targetTimes, maxTileSize, rotate, signal);
    }

    if (grabbed) {
      grabbed.forEach(({ canvas, timeSec }, index) => {
        const name = order[index];
        tiles[name] = canvas;
        const pose = firstPose(landmarker.detect(canvas), canvas.width, canvas.height);
        eventPoses[name] = pose;
        posesRun++;
        if (pose) posesFound++;
        eventTimes[name] = timeSec;
        onProgress?.({ phase: "extracting", done: index + 1, total: EVENTS.length });
      });
    } else {
      // No WebCodecs: grab each frame by playing into its time on the <video>
      // element, visiting positions in time order so playback only moves
      // forward. grabFrameAt keeps this off a stale iOS surface as best it can.
      const { video, release } = await loadVideo(file, signal);
      try {
        for (let index = 0; index < order.length; index++) {
          if (signal?.aborted) throw new Error("Cancelled.");
          const name = order[index];
          const canvas = await grabFrameAt(video, eventTimes[name], maxTileSize, rotate, signal);
          tiles[name] = canvas;
          // Pose the tile rather than the video: angles are scale-invariant, and
          // this way the overlay is already in the tile's coordinates.
          const pose = firstPose(landmarker.detect(canvas), canvas.width, canvas.height);
          eventPoses[name] = pose;
          posesRun++;
          if (pose) posesFound++;
          // Report the frame actually landed on (at or just after the target).
          if (Number.isFinite(video.currentTime)) eventTimes[name] = video.currentTime;
          onProgress?.({ phase: "extracting", done: index + 1, total: EVENTS.length });
        }
      } finally {
        release();
      }
    }

    return {
      durationSec: meta.durationSec,
      width: meta.width,
      height: meta.height,
      posesRun,
      posesFound,
      captureMethod,
      eventTimes,
      eventPoses,
      tiles,
    };
  } catch (err) {
    // A failure should carry the numbers that explain it: with them on
    // screen, a bad clip is distinguishable from a bad app at a glance.
    if (err instanceof Error && clip.width > 0) {
      err.message +=
        ` [clip: ${clip.durationSec.toFixed(1)}s, ${clip.width}×${clip.height}, ` +
        `golfer found in ${posesFound} of ${posesRun} detections]`;
    }
    throw err;
  } finally {
    landmarker.close();
  }
}

/** Joint-angle similarity between two analysed swings. */
export function compareAnalyses(a: SwingAnalysis, b: SwingAnalysis): Similarity {
  return compareEventPoses(a.eventPoses, b.eventPoses);
}
