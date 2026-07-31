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
import { biggestGaps, overallScore, type Similarity } from "./core/angles";
import { EVENTS, type EventName, type Pose } from "./core/constants";
import { analyzeSwing, compareAnalyses, type SwingAnalysis } from "./pipeline";
import type { ModelName } from "./pose/landmarker";
import { resolveCrops, videoAnchor, type CropRect } from "./ui/crop";
import { drawPose } from "./ui/draw";
import type { Rotation } from "./video/decode";

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
 * All 8 positions at a glance, the equivalent of the Python's contact sheet.
 * Tapping one opens it below at full size.
 */
function ContactSheet({
  results,
  selected,
  onSelect,
}: {
  results: Results;
  selected: EventName;
  onSelect: (name: EventName) => void;
}) {
  const { yours, pro, similarity } = results;
  return (
    <div className="sheet">
      {EVENTS.map((name) => {
        const position = similarity?.[name] ?? null;
        return (
          <button
            key={name}
            type="button"
            className={`cell${name === selected ? " is-selected" : ""}`}
            onClick={() => onSelect(name)}
            aria-pressed={name === selected}
          >
            <span className="cell-head">
              <span className="cell-name">{label(name)}</span>
              {position && <ScoreBadge score={position.score} small />}
            </span>
            <span className="cell-frames">
              <FrameTile
                tile={yours.tiles[name]}
                pose={yours.eventPoses[name]}
              />
              {pro && (
                <FrameTile
                  tile={pro.tiles[name]}
                  pose={pro.eventPoses[name]}
                />
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/** The selected position, large. Arrows, swipe, and arrow keys step through. */
function PositionDetail({
  name,
  results,
  onStep,
  align,
  onAlign,
}: {
  name: EventName;
  results: Results;
  onStep: (delta: number) => void;
  align: boolean;
  onAlign: (value: boolean) => void;
}) {
  const { yours, pro, similarity } = results;
  const position = similarity?.[name] ?? null;
  const touch = useRef<{ x: number; y: number } | null>(null);

  // One crop per video, anchored on the median across all 8 poses, so the
  // view stays still while stepping and one bad pose cannot bend it. The
  // window shape is resolved jointly so neither side needs letterboxing.
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
        <FrameTile
          tile={yours.tiles[name]}
          pose={yours.eventPoses[name]}
          crop={yoursCrop}
          caption={`you · ${yours.eventTimes[name].toFixed(2)}s`}
        />
        {pro && (
          <FrameTile
            tile={pro.tiles[name]}
            pose={pro.eventPoses[name]}
            crop={proCrop}
            caption={`reference · ${pro.eventTimes[name].toFixed(2)}s`}
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

  const overall = results?.similarity ? overallScore(results.similarity) : null;

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
              <span className="overall-note">100 = identical joint angles</span>
            </div>
          )}

          <ContactSheet
            results={results}
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
