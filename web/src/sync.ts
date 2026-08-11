/**
 * Builds the aligned frame pairs the sync view scrubs through.
 *
 * One decode per clip, at a fixed budget of phase samples. The budget is the
 * point: keeping every frame is what was ruled out early on (a 3.5s 30fps clip
 * at tile size is well over 100MB a side, and a 240fps slo-mo clip is far
 * worse), and sampling by phase rather than by frame makes the cost independent
 * of clip length and frame rate. Two clips of wildly different tempo and fps
 * still produce exactly `count` pairs.
 *
 * Frames are decoded through WebCodecsClip, never the <video> element: this is
 * the same determinism the rest of the pipeline was moved onto, and playing a
 * clip is what made the phone unusable in the first place.
 */

import { comparePoses, type PositionSimilarity } from "./core/angles";
import type { Pose } from "./core/constants";
import {
  anchorPhases,
  anchorTimes,
  planPhases,
  timeAtPhase,
  type Timing,
} from "./core/warp";
import type { SwingAnalysis } from "./pipeline";
import { createLandmarker, firstPose, type ModelName } from "./pose/landmarker";
import { resolveCrops, videoAnchor, type CropRect } from "./ui/crop";

/** One clip's frame at a sampled phase. */
export interface SyncFrame {
  canvas: HTMLCanvasElement;
  pose: Pose | null;
  timeSec: number;
}

/** The two clips at one moment of the swing. */
export interface SyncPair {
  phase: number;
  yours: SyncFrame;
  pro: SyncFrame;
  /** Similarity at this moment, null where either pose is missing. */
  similarity: PositionSimilarity | null;
}

export interface SyncSequence {
  pairs: SyncPair[];
  /** Phase of each of the 8 positions, for ticks on the scrub. */
  anchorPhases: number[];
  yoursCrop: CropRect | null;
  proCrop: CropRect | null;
  /** Duration of the master clip's swing, for playing back at real tempo. */
  masterDurationSec: number;
}

export interface SyncOptions {
  timing?: Timing;
  count?: number;
  model?: ModelName;
  onProgress?: (done: number, total: number) => void;
  signal?: AbortSignal;
}

/**
 * How many pairs to build.
 *
 * Enough that scrubbing feels continuous rather than steppy, few enough to
 * decode and pose in a couple of seconds. 48 across a swing is roughly a sample
 * every 50ms of a 2.4s swing, finer than the eye tracks at slow playback.
 */
export const SYNC_COUNT = 48;

/**
 * Long side for sync frames.
 *
 * Smaller than the 720 used for tiles: these are shown two-up and scrubbed, so
 * the extra pixels buy nothing and cost linearly. At 480 a portrait frame is
 * around 0.5MB, so 48 pairs is roughly 50MB across both clips, comfortably
 * inside the budget that keeping every frame blew.
 */
export const SYNC_SIZE = 480;

export class SyncUnavailableError extends Error {}

export async function loadSyncSequence(
  yours: SwingAnalysis,
  pro: SwingAnalysis,
  { timing = "yours", count = SYNC_COUNT, model = "full", onProgress, signal }: SyncOptions = {},
): Promise<SyncSequence> {
  if (!yours.clip || !pro.clip) {
    throw new SyncUnavailableError(
      "Stepping through both swings together needs frame-accurate decoding, " +
        "which this browser could not do for one of these clips.",
    );
  }

  const yourAnchors = anchorTimes(yours.eventTimes);
  const proAnchors = anchorTimes(pro.eventTimes);
  // The master clip defines the shared parameterization. Under "yours" that
  // makes your swing advance at its own tempo with the reference bending onto
  // it; under "equal" the choice of master does not matter.
  const phases = anchorPhases(yourAnchors, timing);
  const sampled = planPhases(count);

  const yourTimes = sampled.map((p) => timeAtPhase(yourAnchors, phases, p));
  const proTimes = sampled.map((p) => timeAtPhase(proAnchors, phases, p));

  const total = count * 2;
  let done = 0;
  const bump = () => onProgress?.(++done, total);

  const yourFrames = await yours.clip.grab(yourTimes, SYNC_SIZE, yours.rotation, signal);
  const proFrames = await pro.clip.grab(proTimes, SYNC_SIZE, pro.rotation, signal);

  const landmarker = await createLandmarker("IMAGE", { model });
  try {
    const pose = (canvas: HTMLCanvasElement): Pose | null => {
      const found = firstPose(landmarker.detect(canvas), canvas.width, canvas.height);
      bump();
      return found;
    };

    const yourPosed: SyncFrame[] = yourFrames.map((f) => ({
      canvas: f.canvas,
      timeSec: f.timeSec,
      pose: pose(f.canvas),
    }));
    const proPosed: SyncFrame[] = proFrames.map((f) => ({
      canvas: f.canvas,
      timeSec: f.timeSec,
      pose: pose(f.canvas),
    }));

    const pairs: SyncPair[] = sampled.map((phase, i) => ({
      phase,
      yours: yourPosed[i],
      pro: proPosed[i],
      similarity: comparePoses(yourPosed[i].pose, proPosed[i].pose),
    }));

    // One crop per clip for the whole sequence, anchored on the median across
    // every sampled pose. Computed here rather than reused from the featured
    // view because these frames are decoded at a different size, and a crop
    // measured in tile pixels would be wrong in this coordinate space. The
    // median over all samples is if anything steadier than the 8-pose version.
    const crops = resolveCrops([
      sideFor(yourPosed),
      sideFor(proPosed),
    ]);

    return {
      pairs,
      anchorPhases: phases,
      yoursCrop: crops[0],
      proCrop: crops[1],
      masterDurationSec: Math.max(
        0.1,
        yourAnchors[yourAnchors.length - 1] - yourAnchors[0],
      ),
    };
  } finally {
    landmarker.close();
  }
}

function sideFor(frames: SyncFrame[]): { anchor: ReturnType<typeof videoAnchor>; frameWidth: number } {
  const first = frames[0]?.canvas;
  if (!first) return { anchor: null, frameWidth: 0 };
  return {
    anchor: videoAnchor(frames.map((f) => f.pose), first.height),
    frameWidth: first.width,
  };
}
