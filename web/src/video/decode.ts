/**
 * Sequential frame decoding from a local video file.
 *
 * The project's hardest-won gotcha applies here with more force than it did in
 * Python: **decode sequentially, never seek.** `cv2 CAP_PROP_POS_FRAMES` was
 * off by 2-3 frames on iPhone HEVC and sent an earlier debugging session down a
 * false trail; browser seeking is at least as unreliable. So frames are walked
 * in presentation order with requestVideoFrameCallback and never addressed by
 * index.
 *
 * The cost of that choice is that decoding runs at playback speed, and if the
 * per-frame callback is slower than the frame interval the browser presents
 * fewer frames than the clip contains. That is detectable rather than silent:
 * rVFC reports `presentedFrames`, so gaps are counted and reported back. If
 * drops turn out to matter on device, the fix is WebCodecs (deterministic
 * decode, no timing coupling) at the cost of bundling an MP4 demuxer.
 */

/** Per-frame metadata from requestVideoFrameCallback, narrowed to what we use. */
interface FrameMetadata {
  mediaTime: number;
  presentedFrames: number;
}

export interface WalkOptions {
  /**
   * Playback speed during the walk. Below 1 buys more wall-clock time per
   * frame for slow work, which is the lever for avoiding dropped frames.
   */
  playbackRate?: number;
  /**
   * Called once the video's duration and size are known, before any frame is
   * handed over, so the caller can plan how densely to sample.
   */
  onMetadata?: (info: { durationSec: number; width: number; height: number }) => void;
  signal?: AbortSignal;
}

export interface WalkResult {
  width: number;
  height: number;
  durationSec: number;
  /** Frames actually handed to the callback. */
  frameCount: number;
  /** Frames the browser decoded but never presented to us. */
  droppedFrames: number;
  /** Derived from frame timestamps: the container does not expose a frame rate. */
  fps: number;
}

/**
 * Estimate frame rate from presentation timestamps.
 *
 * There is no browser equivalent of cv2's CAP_PROP_FPS, so it comes from the
 * frames themselves. The median interval is used rather than the mean because
 * a single long gap, from a dropped frame or a variable-frame-rate clip, would
 * drag the mean and quietly distort every frame-rate-derived window.
 */
export function estimateFps(mediaTimesSec: readonly number[]): number {
  if (mediaTimesSec.length < 2) return 30;
  const deltas: number[] = [];
  for (let i = 1; i < mediaTimesSec.length; i++) {
    const d = mediaTimesSec[i] - mediaTimesSec[i - 1];
    if (d > 0) deltas.push(d);
  }
  if (deltas.length === 0) return 30;
  deltas.sort((a, b) => a - b);
  const mid = Math.floor(deltas.length / 2);
  const median =
    deltas.length % 2 === 0 ? (deltas[mid - 1] + deltas[mid]) / 2 : deltas[mid];
  return median > 0 ? 1 / median : 30;
}

/**
 * Scale factor that fits an image inside `maxLongSide` without enlarging it.
 *
 * Retained frames are capped at display resolution: the comparison sheet draws
 * them small anyway, and holding a swing's worth of full-resolution frames is
 * what would run a phone out of memory.
 */
export function fitScale(
  width: number,
  height: number,
  maxLongSide: number,
): number {
  const longSide = Math.max(width, height);
  if (longSide <= maxLongSide) return 1;
  return maxLongSide / longSide;
}

/** Count frames the browser decoded but skipped past, from rVFC's counter. */
export function countDrops(presentedFrames: readonly number[]): number {
  let dropped = 0;
  for (let i = 1; i < presentedFrames.length; i++) {
    const gap = presentedFrames[i] - presentedFrames[i - 1];
    if (gap > 1) dropped += gap - 1;
  }
  return dropped;
}

function once(target: EventTarget, event: string, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const onDone = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Could not load the video (${event} failed).`));
    };
    const onAbort = () => {
      cleanup();
      reject(new Error("Cancelled."));
    };
    const cleanup = () => {
      target.removeEventListener(event, onDone);
      target.removeEventListener("error", onError);
      signal?.removeEventListener("abort", onAbort);
    };
    target.addEventListener(event, onDone, { once: true });
    target.addEventListener("error", onError, { once: true });
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Load a video element from a local file. The file never leaves the browser. */
export async function loadVideo(
  file: Blob,
  signal?: AbortSignal,
): Promise<{ video: HTMLVideoElement; release: () => void }> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.src = url;
  video.muted = true;
  video.playsInline = true;
  // iOS needs the attribute as well as the property to keep playback inline.
  video.setAttribute("playsinline", "");
  video.setAttribute("webkit-playsinline", "");
  video.preload = "auto";
  // Decoding into a canvas taints it unless the source is same-origin; a blob
  // URL is, but being explicit avoids a surprise if this ever moves to a URL.
  video.crossOrigin = "anonymous";

  // Deliberately NOT attached to the document. Attaching it was tried, on the
  // theory that iOS Safari needs the element in the page to present frames
  // reliably, and every variant was worse on desktop: sized down with CSS the
  // browser dropped to a lower decode resolution and moved the detected impact
  // frame by two; placed behind the page or clipped to a small window it
  // stopped being composited and playback died after a single frame. Detached
  // decodes every frame at full resolution here. Revisit only with real
  // numbers off an actual iOS device.
  const release = () => {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  };

  try {
    await once(video, "loadedmetadata", signal);
  } catch (err) {
    release();
    throw err;
  }
  return { video, release };
}

/**
 * Walk every presented frame of a video in order, calling `onFrame` for each.
 *
 * `onFrame` runs synchronously so it finishes before the next frame is
 * presented; anything slow enough to overrun the frame interval shows up as
 * dropped frames in the result rather than as silently missing data.
 */
export async function walkVideoFrames(
  file: Blob,
  onFrame: (video: HTMLVideoElement, index: number, mediaTimeSec: number) => void,
  { playbackRate = 1, onMetadata, signal }: WalkOptions = {},
): Promise<WalkResult> {
  const { video, release } = await loadVideo(file, signal);

  try {
    onMetadata?.({
      durationSec: video.duration,
      width: video.videoWidth,
      height: video.videoHeight,
    });

    if (!("requestVideoFrameCallback" in video)) {
      throw new Error(
        "This browser cannot step through video frames " +
          "(requestVideoFrameCallback is unavailable). Try Safari or Chrome.",
      );
    }

    const mediaTimes: number[] = [];
    const presented: number[] = [];
    let index = 0;

    video.playbackRate = playbackRate;

    await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        video.pause();
        reject(new Error("Cancelled."));
      };
      signal?.addEventListener("abort", onAbort, { once: true });

      /**
       * Resume playback. Pausing on the next frame can abort a still-pending
       * play(), which is expected here rather than a failure, so only real
       * errors are surfaced.
       */
      const resume = () => {
        video.play().catch((err: unknown) => {
          if (err instanceof Error && err.name === "AbortError") return;
          reject(err);
        });
      };

      const step = (_now: number, metadata: FrameMetadata) => {
        if (signal?.aborted) return;
        // Stop the media clock before doing any work. Pose detection takes far
        // longer than a frame interval, and playback does not wait: left
        // running, the browser presents frames we never get a callback for and
        // they are simply lost. Measured on a 27fps clip, that dropped 36 of
        // 57 frames, which then misread the frame rate as 15fps and put every
        // position in the wrong place.
        video.pause();
        try {
          mediaTimes.push(metadata.mediaTime);
          presented.push(metadata.presentedFrames);
          onFrame(video, index, metadata.mediaTime);
          index++;
        } catch (err) {
          reject(err);
          return;
        }
        // Queue the next callback before resuming, so the frame that play()
        // presents cannot arrive before anything is listening for it.
        video.requestVideoFrameCallback(step);
        resume();
      };

      video.requestVideoFrameCallback(step);
      video.addEventListener("ended", () => resolve(), { once: true });
      video.addEventListener(
        "error",
        () => reject(new Error("Video playback failed.")),
        { once: true },
      );
      resume();
    });

    return {
      width: video.videoWidth,
      height: video.videoHeight,
      durationSec: video.duration,
      frameCount: index,
      droppedFrames: countDrops(presented),
      fps: estimateFps(mediaTimes),
    };
  } finally {
    release();
  }
}

/** How long to wait on a single seek before giving up and using what is there. */
const SEEK_TIMEOUT_MS = 3000;

/** Wait for an event, but never indefinitely. Resolves either way. */
function onceWithTimeout(
  target: EventTarget,
  event: string,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      target.removeEventListener(event, done);
      signal?.removeEventListener("abort", done);
      resolve();
    };
    const timer = setTimeout(done, timeoutMs);
    target.addEventListener(event, done, { once: true });
    signal?.addEventListener("abort", done, { once: true });
  });
}

/** One frame fetched by seeking, with the time it actually landed on. */
export interface SampledFrame {
  video: HTMLVideoElement;
  /** Where the seek actually landed, which is not exactly what was asked for. */
  timeSec: number;
  index: number;
}

/**
 * Visit a list of times by seeking, without playing the video.
 *
 * The project's standing rule is to decode sequentially and never seek, and it
 * still holds wherever a specific frame *number* matters: seeking lands a frame
 * or two off, and an earlier debugging session lost hours to exactly that.
 *
 * It does not apply here. This reports the time each seek actually landed on,
 * so nothing depends on hitting an exact frame, and in exchange it escapes the
 * cost that playback imposes: requestVideoFrameCallback only fires while the
 * video plays, so walking a 15 second clip takes 15 seconds however few frames
 * are wanted from it. Seeking to 48 points takes as long as 48 seeks.
 */
export async function sampleAtTimes(
  file: Blob,
  plan: (durationSec: number) => number[],
  onSample: (sample: SampledFrame) => void,
  { signal }: { signal?: AbortSignal } = {},
): Promise<{ width: number; height: number; durationSec: number; samples: number }> {
  const { video, release } = await loadVideo(file, signal);

  try {
    // Metadata alone means duration and dimensions are known but no frame has
    // been decoded yet. Reading pixels now would give a blank one.
    if (video.readyState < 2 /* HAVE_CURRENT_DATA */) {
      await onceWithTimeout(video, "loadeddata", SEEK_TIMEOUT_MS, signal);
    }

    const times = plan(video.duration);
    let index = 0;

    for (const target of times) {
      if (signal?.aborted) throw new Error("Cancelled.");
      // Clamp inside the media: seeking to exactly duration can never resolve.
      const clamped = Math.max(0, Math.min(target, video.duration - 1e-3));

      // Assigning currentTime the value it already holds fires no "seeked"
      // event, so waiting for one hangs forever. Two positions landing on the
      // same time is ordinary, not exotic, on a clip full of freeze frames.
      // Tolerance is well under a frame even at 240fps.
      if (Math.abs(video.currentTime - clamped) > 1e-3) {
        video.currentTime = clamped;
        // Belt and braces: a seek that never reports back must not take the
        // whole analysis down with it. Carrying on gives a slightly wrong
        // frame, which beats hanging with no way out.
        await onceWithTimeout(video, "seeked", SEEK_TIMEOUT_MS, signal);
      }

      onSample({ video, timeSec: video.currentTime, index });
      index++;
    }

    return {
      width: video.videoWidth,
      height: video.videoHeight,
      durationSec: video.duration,
      samples: index,
    };
  } finally {
    release();
  }
}

/**
 * Copy the video's current frame into a new canvas, scaled to fit.
 *
 * Each retained frame gets its own canvas so it survives after playback has
 * moved on; drawing into one shared canvas would leave every tile showing the
 * last frame.
 */
export function captureFrame(
  video: HTMLVideoElement,
  maxLongSide: number,
): HTMLCanvasElement {
  const scale = fitScale(video.videoWidth, video.videoHeight, maxLongSide);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get a 2D canvas context.");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}
