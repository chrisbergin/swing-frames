/**
 * Swing Frames: pick a swing video, get the 8 key positions back, optionally
 * against a second video.
 *
 * Both videos are read straight off the device and analysed in the browser.
 * Nothing is uploaded: no server, no account, and no waiting on range LTE to
 * push a few hundred megabytes of slo-mo somewhere.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import { biggestGaps, overallScore, type Similarity } from "./core/angles";
import { EVENTS, type EventName, type Pose } from "./core/constants";
import { analyzeSwing, compareAnalyses, type SwingAnalysis } from "./pipeline";
import type { ModelName } from "./pose/landmarker";
import { drawPose } from "./ui/draw";

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

/** One extracted frame with its skeleton drawn on. */
function FrameTile({
  tile,
  pose,
  scale,
  caption,
}: {
  tile: HTMLCanvasElement | null;
  pose: Pose | null;
  scale: number;
  caption: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !tile) return;
    canvas.width = tile.width;
    canvas.height = tile.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(tile, 0, 0);
    if (pose) drawPose(ctx, pose, scale);
  }, [tile, pose, scale]);

  return (
    <div className="tile">
      {tile ? <canvas ref={ref} /> : <div className="tile-missing">no frame</div>}
      <span className="tile-caption">{caption}</span>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 80 ? "good" : score >= 60 ? "ok" : "poor";
  return <span className={`score score-${tone}`}>{Math.round(score)}</span>;
}

function PositionRow({
  name,
  yours,
  pro,
  similarity,
}: {
  name: EventName;
  yours: SwingAnalysis;
  pro: SwingAnalysis | null;
  similarity: Similarity | null;
}) {
  const position = similarity?.[name] ?? null;

  return (
    <section className="position">
      <header className="position-head">
        <h3>{name.replace(/_/g, " ")}</h3>
        {position && <ScoreBadge score={position.score} />}
      </header>

      <div className="frames">
        <FrameTile
          tile={yours.tiles[name]}
          pose={yours.poses[yours.events[name]] ?? null}
          scale={yours.tileScale}
          caption={pro ? "you" : `frame ${yours.events[name]}`}
        />
        {pro && (
          <FrameTile
            tile={pro.tiles[name]}
            pose={pro.poses[pro.events[name]] ?? null}
            scale={pro.tileScale}
            caption="reference"
          />
        )}
      </div>

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
  label,
  analysis,
}: {
  label: string;
  analysis: SwingAnalysis;
}) {
  return (
    <li>
      <strong>{label}:</strong> {analysis.frameCount} frames at{" "}
      {analysis.fps.toFixed(1)} fps, {analysis.width}&times;{analysis.height}
      {analysis.droppedFrames > 0 && (
        <span className="warn">, {analysis.droppedFrames} dropped</span>
      )}
    </li>
  );
}

export default function App() {
  const [yourFile, setYourFile] = useState<File | null>(null);
  const [proFile, setProFile] = useState<File | null>(null);
  const [model, setModel] = useState<ModelName>("full");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [results, setResults] = useState<Results | null>(null);

  const run = useCallback(async () => {
    if (!yourFile) return;
    setBusy(true);
    setError("");
    setResults(null);
    const startedAt = performance.now();

    try {
      const analyse = (file: File, label: string) =>
        analyzeSwing(file, {
          model,
          onProgress: ({ phase, done, total }) => {
            const scope = total ? `${done}/${total}` : `${done}`;
            setStatus(
              phase === "tracking"
                ? `Tracking ${label}: ${scope} frames`
                : `Extracting ${label}: ${scope} frames`,
            );
          },
        });

      const yours = await analyse(yourFile, "your swing");
      const pro = proFile ? await analyse(proFile, "the reference") : null;

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
        <label className="picker">
          <span className="picker-label">Your swing</span>
          <input
            type="file"
            accept="video/*"
            disabled={busy}
            onChange={(e) => setYourFile(e.target.files?.[0] ?? null)}
          />
          <span className="filename">{yourFile?.name ?? "no video chosen"}</span>
        </label>

        <label className="picker">
          <span className="picker-label">
            Reference swing <em>optional</em>
          </span>
          <input
            type="file"
            accept="video/*"
            disabled={busy}
            onChange={(e) => setProFile(e.target.files?.[0] ?? null)}
          />
          <span className="filename">{proFile?.name ?? "no video chosen"}</span>
        </label>

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

      <p className="tip">
        Trim the clip to a single swing, starting at address: a clip that opens
        mid-motion gets read as starting at the top of the backswing. Compare
        like camera angle with like.
      </p>

      {results && (
        <>
          {overall !== null && (
            <div className="overall">
              <span className="overall-label">Overall</span>
              <span className="overall-score">{Math.round(overall)}</span>
              <span className="overall-note">100 = identical joint angles</span>
            </div>
          )}

          {EVENTS.map((name) => (
            <PositionRow
              key={name}
              name={name}
              yours={results.yours}
              pro={results.pro}
              similarity={results.similarity}
            />
          ))}

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
              Dropped frames mean the browser played past frames faster than
              they could be analysed. A few are harmless, since the wrist track
              interpolates gaps, but many will move the impact frame.
            </p>
          </details>
        </>
      )}
    </main>
  );
}
