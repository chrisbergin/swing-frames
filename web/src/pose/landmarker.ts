/**
 * MediaPipe PoseLandmarker wrapper.
 *
 * The browser equivalent of get_model() and the landmarker setup in
 * swing_frames.py. Two running modes matter here, exactly as in the Python:
 *
 *   VIDEO  - the main pass over the clip. Carries a temporal prior, which is
 *            what makes it fast, and also what makes it lag the motion-blurred
 *            hands through the hitting zone.
 *   IMAGE  - each frame detected independently, with no prior. Too slow for a
 *            whole clip, but it is what makes the impact re-measure honest.
 */

import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Pose } from "../core/constants";

/** The package declares WasmFileset but does not export it, so derive it. */
type Vision = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

export type ModelName = "lite" | "full" | "heavy";

/** Official model URLs, the same ones the Python downloads. */
export const MODEL_URLS: Record<ModelName, string> = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
  heavy: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
};

export type RunningMode = "VIDEO" | "IMAGE";
export type Delegate = "GPU" | "CPU";

export interface LandmarkerOptions {
  model?: ModelName;
  delegate?: Delegate;
}

/** Anything MediaPipe can read a frame from. */
export type FrameSource = HTMLVideoElement | HTMLCanvasElement | ImageBitmap;

let visionPromise: Promise<Vision> | null = null;

/** Resolve the wasm runtime once and share it across every landmarker. */
function loadVision(): Promise<Vision> {
  // BASE_URL is "/" in dev and "/swing-frames/" on GitHub Pages.
  visionPromise ??= FilesetResolver.forVisionTasks(
    `${import.meta.env.BASE_URL}mediapipe/wasm`,
  );
  return visionPromise;
}

/**
 * Convert MediaPipe's normalized landmarks to pixel coordinates.
 *
 * The JS build reports x and y in 0..1 rather than pixels, so everything
 * downstream, which works in pixels like the Python does, needs them scaled.
 */
export function toPose(
  landmarks: ReadonlyArray<{ x: number; y: number }> | undefined,
  width: number,
  height: number,
): Pose | null {
  if (!landmarks || landmarks.length === 0) return null;
  return landmarks.map((lm) => ({ x: lm.x * width, y: lm.y * height }));
}

/** First detected pose in a result, in pixels, or null if nothing was found. */
export function firstPose(
  result: { landmarks: ReadonlyArray<ReadonlyArray<{ x: number; y: number }>> },
  width: number,
  height: number,
): Pose | null {
  return toPose(result.landmarks[0], width, height);
}

/**
 * Create a landmarker, falling back to CPU if the GPU delegate will not
 * initialise. Mobile Safari's WebGL support is the reason this fallback exists.
 */
export async function createLandmarker(
  runningMode: RunningMode,
  { model = "full", delegate }: LandmarkerOptions = {},
): Promise<PoseLandmarker> {
  const vision = await loadVision();
  const build = (d: Delegate) =>
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URLS[model], delegate: d },
      runningMode,
      numPoses: 1,
    });

  if (delegate) return build(delegate);
  try {
    return await build("GPU");
  } catch {
    return build("CPU");
  }
}

/**
 * Re-measure wrist height over a frame window with independent per-frame
 * detection, for selectImpactFrame() to pick from.
 *
 * The counterpart to the measuring half of refine_impact() in the Python.
 * Only worth running on the small window around impact.
 */
export async function measureWristHeights(
  landmarker: PoseLandmarker,
  frames: ReadonlyArray<FrameSource>,
  startFrame: number,
  height: number,
  wristIndices: readonly [number, number],
): Promise<Array<{ frame: number; y: number }>> {
  const [left, right] = wristIndices;
  const measured: Array<{ frame: number; y: number }> = [];
  for (let i = 0; i < frames.length; i++) {
    const result = landmarker.detect(frames[i]);
    const pts = result.landmarks[0];
    if (!pts) continue;
    measured.push({
      frame: startFrame + i,
      y: ((pts[left].y + pts[right].y) / 2) * height,
    });
  }
  return measured;
}
