/**
 * Time warp between two swings, so both can be walked in step.
 *
 * The 8 detected positions are already a correspondence between the two clips:
 * your top of backswing and the reference's top are the same moment of the same
 * motion, whatever the clock says. So treat them as anchors and map linearly
 * between consecutive ones. One control then drives both clips and alignment is
 * structural rather than eyeballed.
 *
 * Everything here works in PHASE, a number from 0 at address to 1 at finish.
 * Phase is the shared coordinate; each clip converts it to its own time. Phase
 * is not the same as "fraction of the way through the clip", because the two
 * swings have different tempos and that is the whole point.
 *
 * A note on how much the anchors actually buy, because it is easy to overcount:
 * `mid_downswing` and `mid_follow_through` are placed by TIME, at the halfway
 * point of their own phase in each clip (see notes.md, 2026-08-01). A linear map
 * across top -> impact already passes exactly through a midpoint defined that
 * way, so those two anchors are redundant here and contribute no correction.
 * They are kept in the list because including them changes nothing and dropping
 * them would just be a special case to explain. The anchors doing real work are
 * address, toe_up, mid_backswing, top, impact and finish, which means the
 * downswing gets a single linear segment while the backswing gets three.
 */

import { EVENTS, type EventName } from "./constants";

/** How the scrub control maps onto the swing. */
export type Timing =
  /** Your swing plays at its real tempo; the reference is warped onto it. */
  | "yours"
  /** Every segment gets equal travel, stretching the fast downswing out. */
  | "equal";

/** Anchor times for one clip, in swing order. */
export function anchorTimes(times: Record<EventName, number>): number[] {
  return EVENTS.map((name) => times[name]);
}

/**
 * The phase assigned to each anchor: the shared parameterization both clips are
 * read through.
 *
 * Under "equal" the anchors are evenly spread, so each segment takes the same
 * share of the scrub however long it really lasts. Under "yours" they sit at
 * their real fraction of the master clip's duration, so that clip advances at
 * its own tempo and the other one bends.
 *
 * The result is forced non-decreasing and pinned to 0 and 1 at the ends. Event
 * times are normally in order, but a detection can land out of order on an
 * awkward clip and a non-monotonic parameterization would make the scrub jump
 * backwards.
 */
export function anchorPhases(master: readonly number[], timing: Timing): number[] {
  const n = master.length;
  if (n < 2) return [0];
  const even = master.map((_, i) => i / (n - 1));
  if (timing === "equal") return even;

  const first = master[0];
  const span = master[n - 1] - first;
  if (!(span > 0)) return even;

  const phases: number[] = [];
  let floor = 0;
  for (let i = 0; i < n; i++) {
    if (i === n - 1) {
      phases.push(1);
      break;
    }
    const raw = i === 0 ? 0 : (master[i] - first) / span;
    floor = Math.min(1, Math.max(floor, raw));
    phases.push(floor);
  }
  return phases;
}

/**
 * The time in one clip at a given phase, interpolating between its anchors.
 *
 * `phases` is the shared parameterization from anchorPhases; `times` is the
 * clip's own anchor times. Feed one clip's times to get its moment, the other's
 * to get the corresponding moment.
 */
export function timeAtPhase(
  times: readonly number[],
  phases: readonly number[],
  phase: number,
): number {
  const n = Math.min(times.length, phases.length);
  if (n === 0) return 0;
  if (n === 1) return times[0];
  const p = Math.min(1, Math.max(0, phase));

  // Last segment whose start is at or before p, so an exact anchor hit resolves
  // to that anchor rather than to the end of the previous segment.
  let i = 0;
  for (let k = 0; k < n - 1; k++) {
    if (phases[k] <= p) i = k;
  }

  const p0 = phases[i];
  const p1 = phases[i + 1];
  const t0 = times[i];
  const t1 = times[i + 1];
  // Coincident anchors have no interior to interpolate; take the later time so
  // the mapping never runs backwards.
  if (!(p1 > p0)) return Math.max(t0, t1);
  const f = (p - p0) / (p1 - p0);
  return t0 + f * (t1 - t0);
}

/** `count` phase samples spanning address to finish inclusive. */
export function planPhases(count: number): number[] {
  if (count < 2) return [0];
  return Array.from({ length: count }, (_, i) => i / (count - 1));
}

/**
 * Where each named position falls on the scrub, for drawing ticks.
 *
 * Just the anchor phases paired with their names, but it keeps the view from
 * having to know that the anchors and EVENTS are the same list in the same
 * order.
 */
export function anchorMarks(
  phases: readonly number[],
): Array<{ name: EventName; phase: number }> {
  return EVENTS.map((name, i) => ({ name, phase: phases[i] ?? i / (EVENTS.length - 1) }));
}
