/**
 * Swing Frames: pick a swing video, get the 8 key positions back, optionally
 * against a second video.
 *
 * Both videos are read straight off the device and analysed in the browser.
 * Nothing is uploaded: no server, no account, and no waiting on range LTE to
 * push a few hundred megabytes of slo-mo somewhere.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import {
  biggestGaps,
  compareEventPoses,
  overallScore,
  type PositionSimilarity,
  type Similarity,
} from "./core/angles";
import { EVENTS, type EventName, type Pose } from "./core/constants";
import { analyzeSwing, compareAnalyses, type SwingAnalysis } from "./pipeline";
import { createLandmarker, firstPose, type ModelName } from "./pose/landmarker";
import { resolveCrops, videoAnchor, type CropRect } from "./ui/crop";
import { drawPose } from "./ui/draw";
import {
  anyNudged,
  clampOffset,
  emptyNudges,
  frameAt,
  halfFor,
  nextHalf,
  offsetOf,
  posesInPlay,
  setNudge,
  shouldWiden,
  NUDGE_SIZE,
  type NudgeMap,
  type NudgeStore,
  type NudgeWindow,
  type Side,
} from "./ui/nudge";
import type { Rotation } from "./video/decode";

/**
 * Frame stepping around the selected position, per side and on demand.
 *
 * Nothing decodes until the first step for a side, so viewing positions stays
 * instant. The two pieces of state have deliberately different lifetimes:
 *
 *   - The decoded window is dropped when the position changes, so memory stays
 *     bounded to the one position being examined rather than all eight. It is
 *     re-decoded twice as wide when a step reaches its edge.
 *   - The frame stepped to survives until the next analysis, because a nudge is
 *     a correction rather than a peek. The offset stored alongside it is the
 *     single source of truth for where each side sits, so a revisited position
 *     picks up exactly where it was left.
 */
function useNudge(results: Results | null, selected: EventName, model: ModelName) {
  const [windows, setWindows] = useState<Record<Side, NudgeWindow | null>>({
    yours: null,
    pro: null,
  });
  const [nudges, setNudges] = useState<NudgeStore>(emptyNudges);
  const [loading, setLoading] = useState<Record<Side, boolean>>({
    yours: false,
    pro: false,
  });
  const lmRef = useRef<ReturnType<typeof createLandmarker> | null>(null);
  // Guards the decode against overlapping steps. A ref, not the `loading`
  // state: disabling the buttons only takes effect on the next render, so taps
  // landing in the gap (easy on a phone, and the natural way to walk several
  // frames) would each start a full-clip decode. Concurrent decodes peg the
  // main thread and race on setWindows, which can leave a stored offset
  // pointing outside the window that finished last.
  const decodingRef = useRef<Record<Side, boolean>>({ yours: false, pro: false });

  useEffect(() => {
    setWindows({ yours: null, pro: null });
  }, [selected, results]);

  useEffect(() => {
    setNudges(emptyNudges());
  }, [results]);

  const step = useCallback(
    async (side: Side, delta: number) => {
      const analysis = side === "yours" ? results?.yours : (results?.pro ?? null);
      const clip = analysis?.clip;
      if (!analysis || !clip || decodingRef.current[side]) return;

      const offset = offsetOf(nudges[side], selected);
      let win = windows[side];

      // Decode on the first step for this side, and again at twice the width
      // when a step reaches the edge, so a position that needs more than the
      // initial window can still be walked all the way in.
      if (!win || shouldWiden(win, offset, delta)) {
        const half = win ? nextHalf(win) : halfFor(offset);
        decodingRef.current[side] = true;
        setLoading((l) => ({ ...l, [side]: true }));
        try {
          const lm = await (lmRef.current ??= createLandmarker("IMAGE", { model }));
          const { frames, centerIndex } = await clip.framesAround(
            analysis.eventTimes[selected],
            half,
            NUDGE_SIZE,
            analysis.rotation,
          );
          // Posed here rather than on demand: the overlay needs it anyway, and
          // it is what makes the stepped frame scorable for free.
          const decoded: NudgeWindow = {
            frames: frames.map((f) => ({
              canvas: f.canvas,
              timeSec: f.timeSec,
              pose: firstPose(lm.detect(f.canvas), f.canvas.width, f.canvas.height),
            })),
            center: centerIndex,
            half,
          };
          win = decoded;
          setWindows((w) => ({ ...w, [side]: decoded }));
        } finally {
          decodingRef.current[side] = false;
          setLoading((l) => ({ ...l, [side]: false }));
        }
      }

      const w = win;
      if (!w) return;
      const next = clampOffset(w, offset + delta);
      setNudges((n) => ({
        ...n,
        [side]: setNudge(n[side], selected, frameAt(w, next), next),
      }));
    },
    [results, selected, model, windows, nudges],
  );

  return { windows, nudges, loading, step };
}

/** What a position currently shows: the frame stepped to, or the detected one. */
function shownFrame(analysis: SwingAnalysis, map: NudgeMap, name: EventName) {
  const nudge = map[name];
  if (nudge) {
    return {
      tile: nudge.canvas,
      pose: nudge.pose,
      timeSec: nudge.timeSec,
      offset: nudge.offset,
    };
  }
  return {
    tile: analysis.tiles[name],
    pose: analysis.eventPoses[name],
    timeSec: analysis.eventTimes[name],
    offset: 0,
  };
}

const ROTATIONS: Array<[Rotation, string]> = [
  [0, "no rotation"],
  [90, "rotate 90° right"],
  [180, "rotate 180°"],
  [270, "rotate 90° left"],
];

interface Results {
  yours: SwingAnalysis;
  pro: SwingAnalysis | null;
  similarity: Similarity | null;
  elapsedMs: number;
}

const MODEL_LABELS: Record<ModelName, string> = {
  lite: "Lite (fastest)",
  full: "Full (default)",
  heavy: "Heavy (slowest)",
};

const label = (name: EventName) => name.replace(/_/g, " ");

/** Rendered height of an aligned (cropped) tile, in canvas pixels. */
const ALIGNED_TILE_HEIGHT = 720;

/** One extracted frame with its skeleton drawn on, optionally cropped so the
 * golfer sits at a standard size and position. */
function FrameTile({
  tile,
  pose,
  caption,
  crop,
}: {
  tile: HTMLCanvasElement | null;
  pose: Pose | null;
  caption?: string;
  crop?: CropRect | null;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !tile) return;
    const ctx0 = canvas.getContext("2d");
    if (!ctx0) return;
    const ctx = ctx0;

    if (!crop) {
      canvas.width = tile.width;
      canvas.height = tile.height;
      ctx.drawImage(tile, 0, 0);
      // Poses are measured off the tile, so they are already in its coordinates.
      if (pose) drawPose(ctx, pose, 1);
      return;
    }

    const scale = ALIGNED_TILE_HEIGHT / crop.sh;
    canvas.width = Math.round(crop.sw * scale);
    canvas.height = ALIGNED_TILE_HEIGHT;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // The crop may reach past the frame; draw the part that exists and leave
    // the overhang black, so the alignment never bends near an edge.
    const ox = Math.max(crop.sx, 0);
    const oy = Math.max(crop.sy, 0);
    const ox2 = Math.min(crop.sx + crop.sw, tile.width);
    const oy2 = Math.min(crop.sy + crop.sh, tile.height);
    if (ox2 > ox && oy2 > oy) {
      ctx.drawImage(
        tile,
        ox,
        oy,
        ox2 - ox,
        oy2 - oy,
        (ox - crop.sx) * scale,
        (oy - crop.sy) * scale,
        (ox2 - ox) * scale,
        (oy2 - oy) * scale,
      );
    }
    if (pose) {
      ctx.save();
      ctx.setTransform(scale, 0, 0, scale, -crop.sx * scale, -crop.sy * scale);
      drawPose(ctx, pose, 1);
      ctx.restore();
    }
  }, [tile, pose, crop]);

  return (
    <div className="tile">
      {tile ? <canvas ref={ref} /> : <div className="tile-missing">no frame</div>}
      {caption && <span className="tile-caption">{caption}</span>}
    </div>
  );
}

function ScoreBadge({ score, small }: { score: number; small?: boolean }) {
  const tone = score >= 80 ? "good" : score >= 60 ? "ok" : "poor";
  return (
    <span className={`score score-${tone}${small ? " score-small" : ""}`}>
      {Math.round(score)}
    </span>
  );
}

/**
 * The score the detected frames gave, shown once a nudge has moved the live one.
 *
 * A nudged score is the user's judgement and two people will not reproduce it;
 * the detected score is a function of the two clips alone. Keeping it on screen
 * means the reproducible number is never lost behind the corrected one.
 */
function BaselineNote({ position }: { position: PositionSimilarity | null }) {
  return (
    <span className="baseline-note">
      detected {position ? Math.round(position.score) : "n/a"}
    </span>
  );
}

/**
 * All 8 positions at a glance, the equivalent of the Python's contact sheet.
 * Tapping one opens it below at full size.
 *
 * Shows the frames in play, nudges included, and scores them: the sheet is the
 * overview, so it would be misleading for it to keep showing a frame the user
 * has already corrected.
 */
function ContactSheet({
  results,
  similarity,
  nudges,
  selected,
  onSelect,
}: {
  results: Results;
  similarity: Similarity | null;
  nudges: NudgeStore;
  selected: EventName;
  onSelect: (name: EventName) => void;
}) {
  const { yours, pro } = results;
  return (
    <div className="sheet">
      {EVENTS.map((name) => {
        const position = similarity?.[name] ?? null;
        const yourShown = shownFrame(yours, nudges.yours, name);
        const proShown = pro ? shownFrame(pro, nudges.pro, name) : null;
        const nudged = yourShown.offset !== 0 || (proShown?.offset ?? 0) !== 0;
        return (
          <button
            key={name}
            type="button"
            className={`cell${name === selected ? " is-selected" : ""}`}
            onClick={() => onSelect(name)}
            aria-pressed={name === selected}
          >
            <span className="cell-head">
              <span className="cell-name">
                {label(name)}
                {nudged && (
                  <span className="cell-nudged" title="nudged off the detected frame">
                    {" "}
                    &bull;
                  </span>
                )}
              </span>
              {position && <ScoreBadge score={position.score} small />}
            </span>
            <span className="cell-frames">
              <FrameTile tile={yourShown.tile} pose={yourShown.pose} />
              {proShown && <FrameTile tile={proShown.tile} pose={proShown.pose} />}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** One side of the detail: the frame, its caption, and frame-step controls. */
function SideFrame({
  sideLabel,
  analysis,
  name,
  crop,
  nudges,
  window: win,
  loading,
  onStep,
}: {
  sideLabel: string;
  analysis: SwingAnalysis;
  name: EventName;
  crop: CropRect | null;
  nudges: NudgeMap;
  window: NudgeWindow | null;
  loading: boolean;
  onStep: (delta: number) => void;
}) {
  const { tile, pose, timeSec, offset } = shownFrame(analysis, nudges, name);
  const canStep = analysis.clip !== null;
  // A button is dead only when it would change nothing: the window cannot reach
  // further and widening it would not help, meaning the clip itself ran out.
  const stuck = (delta: number) =>
    !!win && !shouldWiden(win, offset, delta) && clampOffset(win, offset + delta) === offset;

  return (
    <div className="side">
      <FrameTile
        tile={tile}
        pose={pose}
        crop={crop}
        caption={`${sideLabel} · ${timeSec.toFixed(2)}s${offset ? ` (${offset > 0 ? "+" : ""}${offset})` : ""}`}
      />
      {canStep && (
        <div className="nudge">
          <button
            type="button"
            className="nudge-btn"
            aria-label="previous frame"
            disabled={loading || stuck(-1)}
            onClick={() => onStep(-1)}
          >
            ‹
          </button>
          <span className="nudge-label">{loading ? "…" : "frame"}</span>
          <button
            type="button"
            className="nudge-btn"
            aria-label="next frame"
            disabled={loading || stuck(1)}
            onClick={() => onStep(1)}
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}

/** The selected position, large. Arrows, swipe, and arrow keys step through. */
function PositionDetail({
  name,
  results,
  similarity,
  baseline,
  nudge,
  onStep,
  align,
  onAlign,
}: {
  name: EventName;
  results: Results;
  /** Scores from the frames on screen, nudges included. */
  similarity: Similarity | null;
  /** Scores from the detected frames, kept visible as the reproducible number. */
  baseline: Similarity | null;
  nudge: ReturnType<typeof useNudge>;
  onStep: (delta: number) => void;
  align: boolean;
  onAlign: (value: boolean) => void;
}) {
  const { yours, pro } = results;
  const position = similarity?.[name] ?? null;
  const touch = useRef<{ x: number; y: number } | null>(null);
  const { windows, nudges, loading, step } = nudge;
  const nudgedHere =
    offsetOf(nudges.yours, name) !== 0 || offsetOf(nudges.pro, name) !== 0;

  // One crop per video, anchored on the median across all 8 DETECTED poses, so
  // the view stays still while stepping (a nudged pose must not move it) and
  // one bad pose cannot bend it. The window shape is resolved jointly so
  // neither side needs letterboxing.
  const [yoursCrop, proCrop] = useMemo<Array<CropRect | null>>(() => {
    if (!align) return [null, null];
    const sideFor = (analysis: SwingAnalysis) => {
      const tile = EVENTS.map((n) => analysis.tiles[n]).find((t) => t !== null);
      if (!tile) return { anchor: null, frameWidth: 0 };
      const poses = EVENTS.map((n) => analysis.eventPoses[n]);
      return { anchor: videoAnchor(poses, tile.height), frameWidth: tile.width };
    };
    const sides = [sideFor(yours)];
    if (pro) sides.push(sideFor(pro));
    const crops = resolveCrops(sides);
    return [crops[0], crops[1] ?? null];
  }, [align, yours, pro]);

  return (
    <section
      className="detail"
      onTouchStart={(e) => {
        touch.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }}
      onTouchEnd={(e) => {
        if (!touch.current) return;
        const dx = e.changedTouches[0].clientX - touch.current.x;
        const dy = e.changedTouches[0].clientY - touch.current.y;
        touch.current = null;
        // A clearly horizontal gesture only, so vertical scrolling never
        // changes the position by accident.
        if (Math.abs(dx) > 48 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
          onStep(dx < 0 ? 1 : -1);
        }
      }}
    >
      <header className="detail-head">
        <button
          type="button"
          className="nav-btn"
          aria-label="previous position"
          onClick={() => onStep(-1)}
        >
          ‹
        </button>
        <div className="detail-title">
          <h2>{label(name)}</h2>
          {position && <ScoreBadge score={position.score} />}
          {nudgedHere && <BaselineNote position={baseline?.[name] ?? null} />}
          <span className="detail-count">
            {EVENTS.indexOf(name) + 1} / {EVENTS.length}
          </span>
        </div>
        <button
          type="button"
          className="nav-btn"
          aria-label="next position"
          onClick={() => onStep(1)}
        >
          ›
        </button>
      </header>

      <div className="frames">
        <SideFrame
          sideLabel="you"
          analysis={yours}
          name={name}
          crop={yoursCrop}
          nudges={nudges.yours}
          window={windows.yours}
          loading={loading.yours}
          onStep={(d) => step("yours", d)}
        />
        {pro && (
          <SideFrame
            sideLabel="reference"
            analysis={pro}
            name={name}
            crop={proCrop}
            nudges={nudges.pro}
            window={windows.pro}
            loading={loading.pro}
            onStep={(d) => step("pro", d)}
          />
        )}
      </div>

      <label className="align-toggle">
        <input
          type="checkbox"
          checked={align}
          onChange={(e) => onAlign(e.target.checked)}
        />
        align golfers (crop both to the same body size, feet on the same line)
      </label>

      {position && (
        <p className="gaps">
          biggest gaps:{" "}
          {biggestGaps(position, 2)
            .map(([joint, deg]) => `${joint.replace(/_/g, " ")} ${Math.round(deg)}°`)
            .join(", ")}
        </p>
      )}
    </section>
  );
}

function Diagnostics({
  label: text,
  analysis,
}: {
  label: string;
  analysis: SwingAnalysis;
}) {
  return (
    <li>
      <strong>{text}:</strong> {analysis.durationSec.toFixed(1)}s,{" "}
      {analysis.width}&times;{analysis.height}, golfer found in{" "}
      {analysis.posesFound} of {analysis.posesRun} detections, capture via{" "}
      {analysis.captureMethod}
    </li>
  );
}

/** File picker plus the rotation override for clips that decode sideways. */
function ClipPicker({
  label: text,
  optional,
  file,
  onFile,
  rotate,
  onRotate,
  disabled,
}: {
  label: string;
  optional?: boolean;
  file: File | null;
  onFile: (file: File | null) => void;
  rotate: Rotation;
  onRotate: (rotation: Rotation) => void;
  disabled: boolean;
}) {
  return (
    <div className="picker-group">
      <label className="picker">
        <span className="picker-label">
          {text} {optional && <em>optional</em>}
        </span>
        <input
          type="file"
          accept="video/*"
          disabled={disabled}
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        <span className="filename">{file?.name ?? "no video chosen"}</span>
      </label>
      {file && (
        <label className="picker">
          <span className="picker-label">Rotation</span>
          <select
            value={rotate}
            disabled={disabled}
            onChange={(e) => onRotate(Number(e.target.value) as Rotation)}
          >
            {ROTATIONS.map(([value, text]) => (
              <option key={value} value={value}>
                {text}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

export default function App() {
  const [yourFile, setYourFile] = useState<File | null>(null);
  const [proFile, setProFile] = useState<File | null>(null);
  const [yourRotate, setYourRotate] = useState<Rotation>(0);
  const [proRotate, setProRotate] = useState<Rotation>(0);
  const [model, setModel] = useState<ModelName>("full");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [results, setResults] = useState<Results | null>(null);
  const [selected, setSelected] = useState<EventName>("address");
  const [align, setAlign] = useState(true);
  const detailRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // The controls sit above a long results page; jump to the output when it
  // lands rather than leaving the user to scroll past the pickers.
  useEffect(() => {
    if (results) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [results]);

  const step = useCallback((delta: number) => {
    setSelected(
      (cur) =>
        EVENTS[(EVENTS.indexOf(cur) + delta + EVENTS.length) % EVENTS.length],
    );
  }, []);

  // Desktop nicety: arrow keys step through the positions too.
  useEffect(() => {
    if (!results) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [results, step]);

  const run = useCallback(async () => {
    if (!yourFile) return;
    setBusy(true);
    setError("");
    setResults(null);
    const startedAt = performance.now();

    try {
      // A failure must say which clip it came from: with two clips in play,
      // an unattributed error leaves the wrong one getting re-shot.
      const analyse = async (file: File, name: string, rotate: Rotation) => {
        try {
          return await analyzeSwing(file, {
            model,
            rotate,
            onProgress: ({ phase, done, total }) => {
              const verb =
                phase === "tracking"
                  ? "Tracking"
                  : phase === "refining"
                    ? "Pinning down impact in"
                    : "Grabbing frames from";
              setStatus(`${verb} ${name}: ${done}/${total}`);
            },
          });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          throw new Error(`Problem with ${name}: ${msg}`);
        }
      };

      const yours = await analyse(yourFile, "your swing", yourRotate);
      const pro = proFile
        ? await analyse(proFile, "the reference", proRotate)
        : null;

      setSelected("address");
      setResults({
        yours,
        pro,
        similarity: pro ? compareAnalyses(yours, pro) : null,
        elapsedMs: performance.now() - startedAt,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      setStatus("");
    }
  }, [yourFile, proFile, model]);

  const nudge = useNudge(results, selected, model);
  const { nudges } = nudge;

  // Scores from the frames actually on screen. results.similarity stays as the
  // baseline: it comes from the detected frames, so it is the number two runs
  // of the same two clips will always agree on.
  const similarity = useMemo(() => {
    if (!results?.pro) return null;
    return compareEventPoses(
      posesInPlay(results.yours.eventPoses, nudges.yours),
      posesInPlay(results.pro.eventPoses, nudges.pro),
    );
  }, [results, nudges]);

  const baseline = results?.similarity ?? null;
  const overall = similarity ? overallScore(similarity) : null;
  const baselineOverall = baseline ? overallScore(baseline) : null;
  const showBaselineOverall =
    anyNudged(nudges) && baselineOverall !== null && baselineOverall !== overall;

  return (
    <main>
      <header className="app-head">
        <h1>Swing Frames</h1>
        <p className="sub">
          The 8 key positions, pulled out of your swing video. Everything runs
          on this device: your videos are never uploaded.
        </p>
      </header>

      <div className="controls">
        <ClipPicker
          label="Your swing"
          file={yourFile}
          onFile={setYourFile}
          rotate={yourRotate}
          onRotate={setYourRotate}
          disabled={busy}
        />

        <ClipPicker
          label="Reference swing"
          optional
          file={proFile}
          onFile={setProFile}
          rotate={proRotate}
          onRotate={setProRotate}
          disabled={busy}
        />

        <label className="picker">
          <span className="picker-label">Model</span>
          <select
            value={model}
            disabled={busy}
            onChange={(e) => setModel(e.target.value as ModelName)}
          >
            {Object.entries(MODEL_LABELS).map(([value, text]) => (
              <option key={value} value={value}>
                {text}
              </option>
            ))}
          </select>
        </label>

        <button type="button" onClick={run} disabled={!yourFile || busy}>
          {busy ? "Analysing…" : "Analyse swing"}
        </button>

        {busy && <p className="status">{status || "Starting…"}</p>}
        {error && <p className="error">{error}</p>}
      </div>

      <details className="tip">
        <summary>What makes a clip work</summary>
        <ul>
          <li>
            <strong>Whole golfer in frame</strong>, head to feet, the entire
            swing. A cropped or heavily zoomed clip finds no golfer at all.
          </li>
          <li>
            <strong>Trimmed to a single swing, starting at address.</strong> A
            clip that opens mid-motion gets read as starting at the top of the
            backswing.
          </li>
          <li>
            <strong>Decent resolution.</strong> A screen recording of someone
            else's video is often too small; prefer a full-resolution download.
          </li>
          <li>
            <strong>Same camera angle on both clips</strong> (down-the-line
            with down-the-line, face-on with face-on), or the similarity score
            means nothing.
          </li>
        </ul>
      </details>

      {results && (
        <div ref={resultsRef}>
          {overall !== null && (
            <div className="overall">
              <span className="overall-label">Overall</span>
              <span className="overall-score">{Math.round(overall)}</span>
              {showBaselineOverall && (
                <span className="baseline-note">
                  detected {Math.round(baselineOverall)}
                </span>
              )}
              <span className="overall-note">100 = identical joint angles</span>
            </div>
          )}

          <ContactSheet
            results={results}
            similarity={similarity}
            nudges={nudges}
            selected={selected}
            onSelect={(name) => {
              setSelected(name);
              // Tapping a cell used to mean scrolling back down by hand;
              // bring the big view along instead.
              detailRef.current?.scrollIntoView({
                behavior: "smooth",
                block: "start",
              });
            }}
          />
          <div ref={detailRef}>
            <PositionDetail
              name={selected}
              results={results}
              similarity={similarity}
              baseline={baseline}
              nudge={nudge}
              onStep={step}
              align={align}
              onAlign={setAlign}
            />
          </div>

          <details className="diagnostics">
            <summary>Diagnostics</summary>
            <ul>
              <li>
                <strong>model:</strong> {model}
              </li>
              <li>
                <strong>elapsed:</strong>{" "}
                {(results.elapsedMs / 1000).toFixed(1)}s
              </li>
              <Diagnostics label="your swing" analysis={results.yours} />
              {results.pro && (
                <Diagnostics label="reference" analysis={results.pro} />
              )}
            </ul>
            <p className="diagnostics-note">
              "Golfer found" well below the detections run means the pose model
              could not see a person in much of the clip: usually cropped
              framing, low resolution, or a sideways video.
            </p>
          </details>
        </div>
      )}
    </main>
  );
}
