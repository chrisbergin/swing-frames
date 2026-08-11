/**
 * The comparison sheet as a single saveable image.
 *
 * The app already draws this on screen: 8 positions left to right, your swing
 * above the reference's. What it could not do is get it out of the browser,
 * which is what you actually want at a range, where the useful outcome is an
 * image in your camera roll or sent to someone rather than a page you looked at
 * once. This is the browser equivalent of the Python prototype's
 * comparison_*.png.
 */

import { EVENTS, type EventName } from "../core/constants";
import { renderFrame, type FrameView } from "./frame";

/** Height each frame is drawn at in the sheet. */
const CELL_HEIGHT = 520;
/** Widest a single column may get, so a landscape clip cannot stretch the sheet. */
const MAX_CELL_WIDTH = 460;
const GAP = 10;
const PAD = 22;
const HEADER = 62;
const LABEL = 30;
const FOOT = 26;

const BG = "#ffffff";
const INK = "#0f1115";
const MUTED = "#6b7280";

export interface SheetRow {
  /** Shown at the left of the row, e.g. "you". */
  label: string;
  frames: Record<EventName, FrameView>;
}

export interface SheetOptions {
  rows: SheetRow[];
  /** Per-position score, omitted when there is nothing to compare against. */
  scores?: Partial<Record<EventName, number>>;
  overall?: number | null;
  title?: string;
}

const pretty = (name: EventName) => name.replace(/_/g, " ");

/**
 * Composite the positions into one image.
 *
 * Frames are rendered through the same drawFrame the on-screen tiles use, so
 * the saved sheet matches what was on screen, align crop and all. Each column
 * is sized from its own frame's aspect, then every column takes the widest, so
 * the grid stays square even if one clip is a different shape.
 */
export function buildComparisonSheet({
  rows,
  scores,
  overall,
  title = "Swing Frames",
}: SheetOptions): HTMLCanvasElement {
  const rendered = rows.map((row) =>
    EVENTS.map((name) => renderFrame(row.frames[name])),
  );

  let cellWidth = 0;
  for (const row of rendered) {
    for (const canvas of row) {
      if (!canvas) continue;
      cellWidth = Math.max(cellWidth, (canvas.width / canvas.height) * CELL_HEIGHT);
    }
  }
  cellWidth = Math.min(Math.round(cellWidth) || 300, MAX_CELL_WIDTH);

  const cols = EVENTS.length;
  const gridWidth = cols * cellWidth + (cols - 1) * GAP;
  const rowHeight = CELL_HEIGHT + LABEL;
  const width = gridWidth + PAD * 2;
  const height =
    HEADER + rows.length * rowHeight + (rows.length - 1) * GAP + FOOT + PAD * 2;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, width, height);
  ctx.textBaseline = "top";

  ctx.fillStyle = INK;
  ctx.font = "600 30px system-ui, sans-serif";
  ctx.fillText(title, PAD, PAD);
  if (overall != null) {
    ctx.font = "700 30px system-ui, sans-serif";
    const text = `Overall ${Math.round(overall)}`;
    ctx.fillText(text, width - PAD - ctx.measureText(text).width, PAD);
  }

  // Position names and scores run along the top, so the columns are readable
  // without having to recognise the swing positions by eye.
  const gridTop = PAD + HEADER;
  ctx.font = "600 19px system-ui, sans-serif";
  EVENTS.forEach((name, col) => {
    const x = PAD + col * (cellWidth + GAP);
    const y = gridTop - LABEL + 4;
    const text = pretty(name);
    ctx.fillStyle = INK;
    ctx.fillText(text, x, y, cellWidth - 48);
    const score = scores?.[name];
    if (score != null) {
      // Immediately after the name, not right-aligned in the cell: a score
      // pushed to the far edge of a wide column sits against the NEXT column's
      // label and reads as belonging to it.
      const at = x + Math.min(ctx.measureText(text).width, cellWidth - 48) + 10;
      ctx.fillStyle = score >= 80 ? "#1f7a4d" : score >= 60 ? "#b4530a" : "#b3261e";
      ctx.fillText(String(Math.round(score)), at, y);
    }
  });

  rendered.forEach((row, r) => {
    const top = gridTop + r * (rowHeight + GAP);
    row.forEach((frame, col) => {
      const x = PAD + col * (cellWidth + GAP);
      ctx.fillStyle = "#000";
      ctx.fillRect(x, top, cellWidth, CELL_HEIGHT);
      if (!frame) return;
      // Fit inside the cell without distorting: a clip whose shape differs from
      // the column gets bars rather than a stretched golfer.
      const scale = Math.min(cellWidth / frame.width, CELL_HEIGHT / frame.height);
      const w = frame.width * scale;
      const h = frame.height * scale;
      ctx.drawImage(frame, x + (cellWidth - w) / 2, top + (CELL_HEIGHT - h) / 2, w, h);
    });
    ctx.fillStyle = MUTED;
    ctx.font = "500 18px system-ui, sans-serif";
    ctx.fillText(rows[r].label, PAD, top + CELL_HEIGHT + 6);
  });

  ctx.fillStyle = MUTED;
  ctx.font = "400 16px system-ui, sans-serif";
  ctx.fillText(
    "chrisbergin.github.io/swing-frames · 100 = identical joint angles",
    PAD,
    height - PAD - 16,
  );

  return canvas;
}

/**
 * JPEG rather than PNG, and this is the right trade here.
 *
 * The sheet is photographic: sixteen video frames, where PNG's lossless
 * encoding buys nothing visible but costs roughly four times the size. At this
 * resolution 0.9 is indistinguishable, and the difference matters because the
 * point of the file is to be texted or dropped in a camera roll. The sheet
 * paints an opaque background, so losing alpha costs nothing either.
 */
const MIME = "image/jpeg";
const QUALITY = 0.9;

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("could not encode the image"))),
      MIME,
      QUALITY,
    );
  });
}

export type SaveOutcome = "shared" | "downloaded";

/**
 * Hand the sheet to the device.
 *
 * On a phone this has to go through the share sheet: an anchor with `download`
 * does not reliably save to Photos in iOS Safari, whereas sharing a file offers
 * "Save Image" directly. Desktop browsers mostly cannot share files, so they
 * fall back to a download. Must be called from a user gesture or the share
 * sheet will not open.
 */
export async function saveSheet(
  canvas: HTMLCanvasElement,
  filename = "swing-comparison.jpg",
): Promise<SaveOutcome> {
  const blob = await toBlob(canvas);
  const file = new File([blob], filename, { type: MIME });

  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
      return "shared";
    } catch (err) {
      // Dismissing the share sheet is a normal outcome, not a failure worth
      // falling back on: downloading behind their back would be a surprise.
      if (err instanceof DOMException && err.name === "AbortError") return "shared";
    }
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  // Revoking immediately can cancel the download in some browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  return "downloaded";
}
