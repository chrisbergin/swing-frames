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
import { anchorMarks, type Timing } from "./core/warp";
import { analyzeSwing, compareAnalyses, type SwingAnalysis } from "./pipeline";
import { loadSyncSequence, type SyncPair, type SyncSequence } from "./sync";
import { createLandmarker, firstPose, type ModelName } from "./pose/landmarker";
import { resolveCrops, videoAnchor, type CropRect } from "./ui/crop";
import { drawFrame } from "./ui/frame";
import { buildComparisonSheet, saveSheet } from "./ui/sheet";
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

/**
 * One crop per video, anchored on the median across all 8 DETECTED poses.
 *
 * Detected rather than nudged, deliberately: the view has to stay still while
 * stepping, so a nudged pose must not move it, and the median stops one bad
 * pose bending the crop. The window shape is resolved jointly across the clips
 * so neither side needs letterboxing.
 */
function eventCrops(
  yours: SwingAnalysis,
  pro: SwingAnalysis | null,
  align: boolean,
): Array<CropRect | null> {
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
    if (canvas) drawFrame(canvas, { tile, pose, crop });
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

  const [yoursCrop, proCrop] = useMemo(
    () => eventCrops(yours, pro, align),
    [align, yours, pro],
  );

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

/**
 * Similarity across the whole swing, drawn under the scrub.
 *
 * The 8 position scores say how much the two swings differ; this says WHERE in
 * the motion it happens, which is the more useful answer. It costs nothing
 * extra: the pairs are posed anyway to draw the skeletons, and a score is 9
 * joint angles off those poses.
 */
function DivergenceCurve({
  pairs,
  marks,
  phase,
  onScrub,
}: {
  pairs: SyncPair[];
  marks: Array<{ name: EventName; phase: number }>;
  phase: number;
  onScrub: (phase: number) => void;
}) {
  const W = 100;
  const H = 34;
  const y = (score: number) => H - (score / 100) * H;

  // Break the line where a pose was missing rather than drawing through it.
  const runs: string[] = [];
  let current: string[] = [];
  for (const p of pairs) {
    if (p.similarity) {
      current.push(`${(p.phase * W).toFixed(2)},${y(p.similarity.score).toFixed(2)}`);
    } else if (current.length) {
      runs.push(current.join(" "));
      current = [];
    }
  }
  if (current.length) runs.push(current.join(" "));

  return (
    <svg
      className="curve"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="similarity across the swing"
      onClick={(e) => {
        const box = e.currentTarget.getBoundingClientRect();
        onScrub(Math.min(1, Math.max(0, (e.clientX - box.left) / box.width)));
      }}
    >
      {marks.map((m) => (
        <line
          key={m.name}
          className="curve-tick"
          x1={m.phase * W}
          x2={m.phase * W}
          y1={0}
          y2={H}
        />
      ))}
      {runs.map((points, i) => (
        <polyline key={i} className="curve-line" points={points} />
      ))}
      <line className="curve-head" x1={phase * W} x2={phase * W} y1={0} y2={H} />
    </svg>
  );
}

/**
 * Both swings walked in step, driven by one scrub.
 *
 * The pairs are built once per timing mode and cached, so scrubbing and
 * playback are pure state changes with no decoding in the loop. Playback is a
 * timer over those cached frames, never video playback: this project already
 * established that playing a clip is what makes a phone unusable.
 */
function SyncView({
  results,
  model,
  align,
  onAlign,
}: {
  results: Results;
  model: ModelName;
  align: boolean;
  onAlign: (value: boolean) => void;
}) {
  const { yours, pro } = results;
  // "equal" by default, which is not the obvious choice but is the right one.
  // Under "yours" the scrub is proportional to real time, and a phone clip
  // typically opens with the golfer stood still over the ball: IMG_5146 spends
  // 1.9s of its 3.1s between address and toe up, so 60% of the scrub is a
  // motionless frame. Equal per phase gives that hold one seventh instead.
  const [timing, setTiming] = useState<Timing>("equal");
  const [sequence, setSequence] = useState<SyncSequence | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(0.25);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!pro) return;
    let cancelled = false;
    const controller = new AbortController();
    setSequence(null);
    setError("");
    setPlaying(false);
    loadSyncSequence(yours, pro, {
      timing,
      model,
      signal: controller.signal,
      onProgress: (done, total) => {
        if (!cancelled) setStatus(`Lining the swings up: ${done}/${total}`);
      },
    })
      .then((seq) => {
        if (cancelled) return;
        setSequence(seq);
        setIndex(0);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setStatus("");
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [yours, pro, timing, model]);

  // Playback walks the cached pairs on a timer. Under "yours" the master clip
  // is linear in phase, so an even timer reproduces its real tempo scaled by
  // the speed control.
  useEffect(() => {
    if (!playing || !sequence) return;
    const steps = sequence.pairs.length - 1;
    if (steps < 1) return;
    const stepMs = (sequence.masterDurationSec * 1000) / steps / speed;
    const id = setInterval(
      () => setIndex((i) => Math.min(i + 1, steps)),
      Math.max(16, stepMs),
    );
    return () => clearInterval(id);
  }, [playing, sequence, speed]);

  // Stopping at the end lives here rather than inside the index updater: an
  // updater must be pure, and setting other state from within one leaves the
  // timer firing against a value that never changes.
  useEffect(() => {
    if (playing && sequence && index >= sequence.pairs.length - 1) setPlaying(false);
  }, [playing, sequence, index]);

  if (!pro) return null;
  if (error) return <p className="error">{error}</p>;
  if (!sequence) return <p className="status">{status || "Lining the swings up…"}</p>;

  const pair = sequence.pairs[Math.min(index, sequence.pairs.length - 1)];
  const marks = anchorMarks(sequence.anchorPhases);
  const scrubTo = (phase: number) => {
    const steps = sequence.pairs.length - 1;
    setIndex(Math.round(phase * steps));
    setPlaying(false);
  };
  // The position whose anchor the playhead has most recently passed, so the
  // scrub can say where in the swing you are without inventing new names.
  const nearest = marks.reduce((best, m) =>
    Math.abs(m.phase - pair.phase) < Math.abs(best.phase - pair.phase) ? m : best,
  );

  return (
    <section className="sync">
      <div className="frames">
        <FrameTile
          tile={pair.yours.canvas}
          pose={pair.yours.pose}
          crop={align ? sequence.yoursCrop : null}
          caption={`you · ${pair.yours.timeSec.toFixed(2)}s`}
        />
        <FrameTile
          tile={pair.pro.canvas}
          pose={pair.pro.pose}
          crop={align ? sequence.proCrop : null}
          caption={`reference · ${pair.pro.timeSec.toFixed(2)}s`}
        />
      </div>

      <div className="scrub">
        <DivergenceCurve
          pairs={sequence.pairs}
          marks={marks}
          phase={pair.phase}
          onScrub={scrubTo}
        />
        <input
          type="range"
          min={0}
          max={sequence.pairs.length - 1}
          value={index}
          aria-label="scrub through the swing"
          onChange={(e) => {
            setIndex(Number(e.target.value));
            setPlaying(false);
          }}
        />
        <div className="scrub-read">
          <span className="scrub-where">{label(nearest.name)}</span>
          {pair.similarity && <ScoreBadge score={pair.similarity.score} small />}
        </div>
      </div>

      <div className="sync-controls">
        <button
          type="button"
          className="nudge-btn"
          onClick={() => {
            if (index >= sequence.pairs.length - 1) setIndex(0);
            setPlaying((p) => !p);
          }}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <label className="picker">
          <span className="picker-label">Speed</span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={0.15}>Very slow</option>
            <option value={0.25}>Slow</option>
            <option value={0.5}>Half speed</option>
            <option value={1}>Real time</option>
          </select>
        </label>
        <label className="picker">
          <span className="picker-label">Timing</span>
          <select value={timing} onChange={(e) => setTiming(e.target.value as Timing)}>
            <option value="equal">Even per phase</option>
            <option value="yours">Your tempo</option>
          </select>
        </label>
      </div>

      <label className="align-toggle">
        <input
          type="checkbox"
          checked={align}
          onChange={(e) => onAlign(e.target.checked)}
        />
        align golfers (crop both to the same body size, feet on the same line)
      </label>

      <p className="sync-note">
        Both clips are held at the same moment of the swing, not the same clock
        time, by mapping between the 8 detected positions.{" "}
        {timing === "equal"
          ? "Even per phase gives each segment the same share of the scrub: the downswing stretches out, and a long stand-still at address does not eat the whole bar."
          : "Your tempo plays your swing at its real speed and bends the reference onto it. Expect a clip that opens with a long address hold to spend most of the scrub on it."}
      </p>
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
  const [mode, setMode] = useState<"frames" | "sync">("frames");
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

  // Desktop nicety: arrow keys step through the positions too. Not in sync
  // mode, where the arrows belong to the scrub slider.
  useEffect(() => {
    if (!results || mode !== "frames") return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") step(-1);
      if (e.key === "ArrowRight") step(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [results, step, mode]);

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

  // Saving the sheet: the frames in play (nudges included), cropped the way
  // they are on screen, composited into one image. On a phone this opens the
  // share sheet, which is the only reliable route into the camera roll.
  const [saving, setSaving] = useState("");
  const saveComparison = useCallback(async () => {
    if (!results) return;
    setSaving("Building the image…");
    try {
      const [yoursCrop, proCrop] = eventCrops(results.yours, results.pro, align);
      const framesFor = (analysis: SwingAnalysis, map: NudgeMap, crop: CropRect | null) =>
        Object.fromEntries(
          EVENTS.map((name) => {
            const shown = shownFrame(analysis, map, name);
            return [name, { tile: shown.tile, pose: shown.pose, crop }];
          }),
        ) as Record<EventName, { tile: HTMLCanvasElement | null; pose: Pose | null; crop: CropRect | null }>;

      const rows = [
        { label: "you", frames: framesFor(results.yours, nudges.yours, yoursCrop) },
      ];
      if (results.pro) {
        rows.push({
          label: "reference",
          frames: framesFor(results.pro, nudges.pro, proCrop),
        });
      }

      const scores = Object.fromEntries(
        EVENTS.map((name) => [name, similarity?.[name]?.score]).filter(
          ([, v]) => v != null,
        ),
      ) as Partial<Record<EventName, number>>;

      const outcome = await saveSheet(
        buildComparisonSheet({ rows, scores, overall }),
      );
      setSaving(outcome === "shared" ? "" : "Saved to your downloads.");
    } catch (err) {
      setSaving(err instanceof Error ? err.message : String(err));
    }
  }, [results, align, nudges, similarity, overall]);

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

          {results.pro && (
            <div className="modes" role="tablist">
              {(["frames", "sync"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="tab"
                  aria-selected={mode === m}
                  className={`mode${mode === m ? " is-on" : ""}`}
                  onClick={() => setMode(m)}
                >
                  {m === "frames" ? "Frames" : "Full playback"}
                </button>
              ))}
            </div>
          )}

          {mode === "sync" && results.pro ? (
            <SyncView
              results={results}
              model={model}
              align={align}
              onAlign={setAlign}
            />
          ) : (
          <>
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
          <div className="sheet-actions">
            {/* "Save comparison image" read as saving the single frame on
                screen; naming the panel says it is all 8 positions at once. */}
            <button type="button" className="nudge-btn" onClick={saveComparison}>
              Save panel comparison
            </button>
            {saving && <span className="sheet-saving">{saving}</span>}
          </div>

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
          </>
          )}

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
