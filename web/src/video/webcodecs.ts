/**
 * Frame-accurate extraction with WebCodecs, bypassing the <video> element.
 *
 * The <video> path (grabFrameAt in decode.ts) is at the mercy of the browser's
 * compositor: on iOS Safari a seeked, paused, detached element can keep showing
 * a stale frame, and playing into a target drifts the clock. WebCodecs sidesteps
 * all of it: the container is demuxed to encoded samples, a VideoDecoder turns
 * them into frames with exact timestamps, and the frame at each target time is
 * drawn straight to a canvas. Deterministic, and identical on desktop and phone,
 * so desktop testing actually predicts the phone.
 *
 * This is the preferred path; the pipeline falls back to the <video> capture
 * where WebCodecs or the demux is unavailable, so nothing here can regress a
 * browser it does not run on.
 */

import { createFile, DataStream, Endianness } from "mp4box";
import { drawImageRotated, fitScale, rotatedSize, type Rotation } from "./decode";

/** Whether this browser can decode frames without a <video> element. */
export function webCodecsSupported(): boolean {
  return (
    typeof VideoDecoder !== "undefined" &&
    typeof EncodedVideoChunk !== "undefined" &&
    typeof VideoFrame !== "undefined"
  );
}

// The mp4box fork's own types are heavily generic; a narrow local view of the
// handful of fields and callbacks used here keeps this file readable.
interface Mp4Sample {
  data?: Uint8Array;
  cts: number;
  dts: number;
  timescale: number;
  is_sync: boolean;
}

interface Mp4VideoTrack {
  id: number;
  codec: string;
  video?: { width: number; height: number };
  track_width: number;
  track_height: number;
  matrix?: ArrayLike<number>;
}

/**
 * Display rotation from a track's transformation matrix.
 *
 * iPhone clips are recorded sensor-landscape and carry a rotation in the
 * container that a <video> element applies automatically but a raw decoded
 * VideoFrame does not. The matrix's a,b (indices 0,1, in 16.16 fixed point)
 * give the angle; snap it to the nearest quarter turn.
 */
export function rotationFromMatrix(matrix?: ArrayLike<number>): Rotation {
  if (!matrix || matrix.length < 2) return 0;
  const deg = (Math.round(Math.atan2(matrix[1], matrix[0]) * (180 / Math.PI)) + 360) % 360;
  const snapped = Math.round(deg / 90) * 90;
  return (snapped % 360) as Rotation;
}

interface Mp4File {
  onError: (e: string) => void;
  onReady: (info: { videoTracks?: Mp4VideoTrack[] }) => void;
  onSamples: (id: number, user: unknown, samples: Mp4Sample[]) => void;
  setExtractionOptions: (id: number, user: unknown, opts: { nbSamples: number }) => void;
  start: () => void;
  appendBuffer: (buffer: ArrayBuffer) => number;
  flush: () => void;
  getTrackById: (id: number) => unknown;
}

interface DemuxSample {
  data: Uint8Array;
  /** Presentation time (composition), which is what a target matches against. */
  presentationSec: number;
  /** Decode time, which is the order chunks must be fed to the decoder. */
  decodeSec: number;
  isKey: boolean;
}

interface Demuxed {
  config: VideoDecoderConfig;
  samples: DemuxSample[];
  /** Rotation the container asks for, which a raw VideoFrame lacks. */
  containerRotation: Rotation;
}

/**
 * The avcC / hvcC / av1C box bytes a VideoDecoder needs as its `description`.
 *
 * mp4box parses it into the sample-description box tree; serialise that box and
 * drop its 8-byte header to get the raw codec-private data the decoder wants.
 */
function codecDescription(file: Mp4File, trackId: number): Uint8Array {
  // The fork's box tree is loosely typed; reach through it directly.
  const trak = file.getTrackById(trackId) as {
    mdia: { minf: { stbl: { stsd: { entries: Array<Record<string, unknown>> } } } };
  };
  for (const entry of trak.mdia.minf.stbl.stsd.entries) {
    const box = (entry.avcC ?? entry.hvcC ?? entry.hev1 ?? entry.av1C ?? entry.vpcC) as
      | { write(stream: DataStream): void }
      | undefined;
    if (box) {
      const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
      box.write(stream);
      // Drop the 8-byte box header, leaving the raw codec-private data.
      return new Uint8Array(stream.buffer, 8);
    }
  }
  throw new Error("no codec description in track");
}

/** Demux a whole clip into encoded samples plus the decoder config. */
async function demux(file: Blob): Promise<Demuxed> {
  const mp4 = createFile() as unknown as Mp4File;
  const samples: DemuxSample[] = [];
  let containerRotation: Rotation = 0;

  const ready = new Promise<VideoDecoderConfig>((resolve, reject) => {
    mp4.onError = (e) => reject(new Error(`demux failed: ${e}`));
    mp4.onReady = (info) => {
      const track = info.videoTracks?.[0];
      if (!track) {
        reject(new Error("no video track in clip"));
        return;
      }
      containerRotation = rotationFromMatrix(track.matrix);
      const description = codecDescription(mp4, track.id);
      resolve({
        codec: track.codec,
        codedWidth: track.video?.width ?? track.track_width,
        codedHeight: track.video?.height ?? track.track_height,
        description,
      });
      mp4.setExtractionOptions(track.id, null, { nbSamples: Number.POSITIVE_INFINITY });
      mp4.start();
    };
    mp4.onSamples = (_id, _user, list) => {
      for (const s of list) {
        if (!s.data) continue;
        samples.push({
          data: s.data,
          presentationSec: s.cts / s.timescale,
          decodeSec: s.dts / s.timescale,
          isKey: s.is_sync,
        });
      }
    };
  });

  const buffer = (await file.arrayBuffer()) as ArrayBuffer & { fileStart?: number };
  buffer.fileStart = 0;
  mp4.appendBuffer(buffer);
  mp4.flush();

  const config = await ready;
  if (samples.length === 0) throw new Error("no video samples in clip");
  // Feed the decoder in DECODE order: with B-frames, presentation order differs
  // and out-of-order chunks corrupt decoding (intermittent distorted frames).
  samples.sort((a, b) => a.decodeSec - b.decodeSec);
  return { config, samples, containerRotation };
}

/** A captured frame with the exact time of the decoded frame behind it. */
export interface ExtractedFrame {
  canvas: HTMLCanvasElement;
  timeSec: number;
}

function drawVideoFrame(
  frame: VideoFrame,
  maxLongSide: number,
  rotation: Rotation,
): HTMLCanvasElement {
  const w = frame.displayWidth;
  const h = frame.displayHeight;
  const canvas = document.createElement("canvas");
  drawImageRotated(canvas, frame, w, h, rotation, fitScale(w, h, maxLongSide));
  return canvas;
}

/**
 * A demuxed clip that can decode frames on demand without a <video> element.
 *
 * Demuxing happens once at open; each decode walks the encoded samples through
 * a fresh VideoDecoder. Because it never touches the compositor, the frame at a
 * given time is the same on every device, so detection run against it (not just
 * the displayed frames) comes out identical on desktop and phone. That is what
 * fixes positions like impact drifting on the phone: the <video> + MediaPipe
 * detection path is device-dependent, this is not.
 */
export class WebCodecsClip {
  private readonly config: VideoDecoderConfig;
  private readonly samples: DemuxSample[];
  private readonly containerRotation: Rotation;
  /** Presentation time span, for clamping requested times into the clip. */
  readonly firstSec: number;
  readonly lastSec: number;

  private constructor(
    config: VideoDecoderConfig,
    samples: DemuxSample[],
    containerRotation: Rotation,
    firstSec: number,
    lastSec: number,
  ) {
    this.config = config;
    this.samples = samples;
    this.containerRotation = containerRotation;
    this.firstSec = firstSec;
    this.lastSec = lastSec;
  }

  static async open(file: Blob): Promise<WebCodecsClip> {
    const { config, samples, containerRotation } = await demux(file);
    const support = await VideoDecoder.isConfigSupported(config);
    if (!support.supported) throw new Error(`codec unsupported: ${config.codec}`);
    let first = Number.POSITIVE_INFINITY;
    let last = Number.NEGATIVE_INFINITY;
    for (const s of samples) {
      if (s.presentationSec < first) first = s.presentationSec;
      if (s.presentationSec > last) last = s.presentationSec;
    }
    return new WebCodecsClip(config, samples, containerRotation, first, last);
  }

  get durationSec(): number {
    return this.lastSec;
  }

  /** Displayed pixel size after the container's own rotation (not the user's),
   * matching what a <video> element reports. */
  get displayWidth(): number {
    return rotatedSize(
      this.config.codedWidth ?? 0,
      this.config.codedHeight ?? 0,
      this.containerRotation,
    ).width;
  }

  get displayHeight(): number {
    return rotatedSize(
      this.config.codedWidth ?? 0,
      this.config.codedHeight ?? 0,
      this.containerRotation,
    ).height;
  }

  private rotationFor(userRotation: Rotation): Rotation {
    return ((this.containerRotation + userRotation) % 360) as Rotation;
  }

  /** Decode the whole clip once, invoking `onFrame` with each frame's time. */
  private async decodeAll(
    onFrame: (frame: VideoFrame, timeSec: number) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    let decodeError: Error | null = null;
    const decoder = new VideoDecoder({
      output: (frame) => {
        try {
          onFrame(frame, frame.timestamp / 1e6);
        } finally {
          frame.close();
        }
      },
      error: (e) => {
        decodeError = e instanceof Error ? e : new Error(String(e));
      },
    });
    try {
      decoder.configure(this.config);
      for (const s of this.samples) {
        if (signal?.aborted) throw new Error("Cancelled.");
        if (decodeError) throw decodeError;
        // Fed in decode order (samples are sorted by DTS); the chunk timestamp
        // is the presentation time so decoded frames carry it for matching.
        decoder.decode(
          new EncodedVideoChunk({
            type: s.isKey ? "key" : "delta",
            timestamp: Math.round(s.presentationSec * 1e6),
            data: s.data,
          }),
        );
      }
      await decoder.flush();
    } finally {
      if (decoder.state !== "closed") decoder.close();
    }
    if (decodeError) throw decodeError;
  }

  /**
   * The frame at each requested time, for display.
   *
   * For every target the frame kept is the earliest at or after it (matching
   * how impact is defined). Targets are clamped inside the clip so each always
   * resolves to a real frame.
   */
  async grab(
    times: readonly number[],
    maxLongSide: number,
    rotation: Rotation = 0,
    signal?: AbortSignal,
  ): Promise<ExtractedFrame[]> {
    const rot = this.rotationFor(rotation);
    const targets = times.map((t) => Math.max(this.firstSec, Math.min(t, this.lastSec)));
    const canvases: (HTMLCanvasElement | null)[] = targets.map(() => null);
    const chosenTs = targets.map(() => Number.POSITIVE_INFINITY);

    await this.decodeAll((frame, t) => {
      for (let i = 0; i < targets.length; i++) {
        if (t >= targets[i] - 1e-4 && t < chosenTs[i]) {
          canvases[i] = drawVideoFrame(frame, maxLongSide, rot);
          chosenTs[i] = t;
        }
      }
    }, signal);

    return targets.map((t, i) => {
      const canvas = canvases[i];
      if (!canvas) throw new Error(`no frame decoded for ${t.toFixed(3)}s`);
      return { canvas, timeSec: Number.isFinite(chosenTs[i]) ? chosenTs[i] : t };
    });
  }

  /**
   * Draw every frame whose time falls in [loSec, hiSec] and hand it to `onFrame`.
   *
   * Used to re-measure a window deterministically (e.g. narrowing impact): the
   * caller poses each canvas. A modest `maxLongSide` keeps it quick; pose angles
   * and wrist height are scale-invariant.
   */
  async forEachFrameInWindow(
    loSec: number,
    hiSec: number,
    maxLongSide: number,
    onFrame: (canvas: HTMLCanvasElement, timeSec: number) => void,
    rotation: Rotation = 0,
    signal?: AbortSignal,
  ): Promise<void> {
    const rot = this.rotationFor(rotation);
    await this.decodeAll((frame, t) => {
      if (t >= loSec - 1e-4 && t <= hiSec + 1e-4) {
        onFrame(drawVideoFrame(frame, maxLongSide, rot), t);
      }
    }, signal);
  }
}
