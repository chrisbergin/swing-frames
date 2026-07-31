/**
 * Frame access for a local video file.
 *
 * Frames are fetched by seeking (`sampleAtTimes`), never by playing the clip:
 * requestVideoFrameCallback only fires during playback, so walking a clip costs
 * its full running time however few frames are wanted from it. The project's
 * standing "never seek" rule is about frame *numbers*: every seek here reports
 * the time it actually landed on, and nothing depends on hitting an exact
 * frame.
 */

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
    // iOS Safari hands the canvas black frames from a video that has never
    // played, however long it has been seeked and waited on. One muted
    // play/pause primes the decoder. Autoplay refusal just means no priming,
    // so it is not an error worth surfacing.
    try {
      const playing = video.play();
      if (playing) {
        await Promise.race([
          playing,
          new Promise((r) => setTimeout(r, 300)),
        ]);
      }
    } catch {
      /* see above */
    }
    video.pause();
  } catch (err) {
    release();
    throw err;
  }
  return { video, release };
}

/**
 * Give the decoder a beat to land pixels after a seek.
 *
 * The "seeked" event only says the media clock moved: on iOS Safari the frame
 * itself can arrive at the canvas after it. This wait is paid only when a
 * capture actually came back blank, never on the happy path: the video element
 * is detached, therefore never composited, therefore
 * requestVideoFrameCallback never fires for it and cannot be awaited instead.
 */
export function settleDelay(ms = 200): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Coarse 8x8 luminance signature of a canvas, for cheap comparisons. */
export function probeSignature(canvas: HTMLCanvasElement): number[] {
  const probe = document.createElement("canvas");
  probe.width = 8;
  probe.height = 8;
  const ctx = probe.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];
  ctx.drawImage(canvas, 0, 0, 8, 8);
  const { data } = ctx.getImageData(0, 0, 8, 8);
  const sig: number[] = [];
  for (let i = 0; i < data.length; i += 4) {
    sig.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
  }
  return sig;
}

/**
 * Whether a canvas holds (near-)nothing but black pixels.
 *
 * A real video frame always has some pixel above the threshold, even at night
 * at the range.
 */
export function isBlankCanvas(canvas: HTMLCanvasElement): boolean {
  return probeSignature(canvas).every((v) => v <= 16);
}

/** Mean absolute difference between two probe signatures. */
export function signatureDiff(a: readonly number[], b: readonly number[]): number {
  if (a.length === 0 || a.length !== b.length) return Infinity;
  let total = 0;
  for (let i = 0; i < a.length; i++) total += Math.abs(a[i] - b[i]);
  return total / a.length;
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
  onSample: (sample: SampledFrame) => void | Promise<void>,
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

      await onSample({ video, timeSec: video.currentTime, index });
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
 * Clockwise rotation applied to every frame before it is used.
 *
 * The port of the Python's --rotate. Phone videos carry rotation metadata the
 * browser honours, so this exists for clips without it: screen recordings and
 * some downloads, which otherwise decode sideways and pose as "no golfer".
 */
export type Rotation = 0 | 90 | 180 | 270;

/** Canvas size holding a `width` x `height` frame after rotation and scaling. */
export function rotatedSize(
  width: number,
  height: number,
  rotation: Rotation,
  scale = 1,
): { width: number; height: number } {
  const w = Math.round(width * scale);
  const h = Math.round(height * scale);
  const swapped = rotation === 90 || rotation === 270;
  return { width: swapped ? h : w, height: swapped ? w : h };
}

/** Draw the video's current frame into `canvas`, rotated clockwise and scaled. */
export function drawRotatedFrame(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  rotation: Rotation,
  scale = 1,
): void {
  const w = Math.round(video.videoWidth * scale);
  const h = Math.round(video.videoHeight * scale);
  const size = rotatedSize(video.videoWidth, video.videoHeight, rotation, scale);
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get a 2D canvas context.");
  ctx.save();
  if (rotation === 90) {
    ctx.translate(canvas.width, 0);
    ctx.rotate(Math.PI / 2);
  } else if (rotation === 180) {
    ctx.translate(canvas.width, canvas.height);
    ctx.rotate(Math.PI);
  } else if (rotation === 270) {
    ctx.translate(0, canvas.height);
    ctx.rotate(-Math.PI / 2);
  }
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
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
  rotation: Rotation = 0,
): HTMLCanvasElement {
  const scale = fitScale(video.videoWidth, video.videoHeight, maxLongSide);
  const canvas = document.createElement("canvas");
  drawRotatedFrame(canvas, video, rotation, scale);
  return canvas;
}

/**
 * Capture the current frame, waiting out presentation glitches.
 *
 * Reading right after "seeked" is usually correct but not always: measured
 * against sequential ground truth, an occasional capture holds a frame
 * several frames away from where the seek landed (a different position each
 * run), and on iOS a never-settled surface reads black. So capture until two
 * consecutive reads agree: a settled frame converges immediately, a glitched
 * one gets replaced by what the surface settles on.
 */
export async function captureSettledFrame(
  video: HTMLVideoElement,
  maxLongSide: number,
  rotation: Rotation = 0,
): Promise<HTMLCanvasElement> {
  let tile = captureFrame(video, maxLongSide, rotation);
  for (let attempt = 0; attempt < 3 && isBlankCanvas(tile); attempt++) {
    await settleDelay();
    tile = captureFrame(video, maxLongSide, rotation);
  }
  let sig = probeSignature(tile);
  for (let attempt = 0; attempt < 3; attempt++) {
    await settleDelay(40);
    const next = captureFrame(video, maxLongSide, rotation);
    const nextSig = probeSignature(next);
    const settled = signatureDiff(sig, nextSig) < 1;
    tile = next;
    sig = nextSig;
    if (settled) break;
  }
  return tile;
}
