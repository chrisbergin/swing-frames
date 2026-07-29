/**
 * Parity against the Python reference implementation.
 *
 * The fixtures are real output from swing_frames.py on the project's two test
 * clips, captured by tools/dump_parity_fixture.py: the per-frame wrist
 * midpoints going in, and the smoothed track and detected positions coming
 * out. Feeding the port the same input must reproduce the same positions.
 *
 * Between them the clips cover the two video shapes the detector has to
 * handle: continuous phone footage, and a pause-and-step analysis edit.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it, expect } from "vitest";
import { detectEvents, wristTrack } from "./events";
import { EVENTS, LM, type EventName, type Pose } from "./constants";

interface Fixture {
  video: string;
  fps: number;
  n_frames: number;
  detected_frames: number;
  wrists: Array<[number, number] | null>;
  smoothed_ys: number[];
  events: Record<EventName, number>;
}

const here = dirname(fileURLToPath(import.meta.url));

function loadFixture(name: string): Fixture {
  return JSON.parse(
    readFileSync(join(here, "__fixtures__", `${name}.json`), "utf8"),
  ) as Fixture;
}

/**
 * Rebuild poses from the recorded wrist midpoints. Both wrists are placed on
 * the midpoint, which averages back to that same midpoint: the tracker reads
 * nothing else, so this is exactly the input the Python saw.
 */
function toPoses(fixture: Fixture): Array<Pose | null> {
  return fixture.wrists.map((w) => {
    if (!w) return null;
    const pts: Pose = Array.from({ length: 33 }, () => ({ x: 0, y: 0 }));
    pts[LM.L_WRIST] = { x: w[0], y: w[1] };
    pts[LM.R_WRIST] = { x: w[0], y: w[1] };
    return pts;
  });
}

describe.each([
  ["IMG_5146", "continuous phone footage, driver, down the line"],
  ["Grant-Horvat-Driver2", "pause-and-step analysis edit with freeze frames"],
])("%s (%s)", (name) => {
  const fixture = loadFixture(name);
  const poses = toPoses(fixture);

  it("reproduces the smoothed wrist track", () => {
    const { ys } = wristTrack(poses, fixture.fps);
    expect(ys).toHaveLength(fixture.smoothed_ys.length);
    for (let i = 0; i < ys.length; i++) {
      expect(ys[i]).toBeCloseTo(fixture.smoothed_ys[i], 4);
    }
  });

  it("reproduces every detected swing position exactly", () => {
    const { ys } = wristTrack(poses, fixture.fps);
    const events = detectEvents(ys, fixture.fps);
    expect(events).toEqual(fixture.events);
  });

  it("returns positions in swing order", () => {
    const { ys } = wristTrack(poses, fixture.fps);
    const events = detectEvents(ys, fixture.fps);
    const frames = EVENTS.map((e) => events[e]);
    expect(frames.every((v, i) => i === 0 || v > frames[i - 1])).toBe(true);
  });
});
