"""Extract key golf swing position frames from a video using MediaPipe Pose.

Point it at a video of a single swing (trimmed so the swing is the main motion
in the clip). It tracks the wrists through the video and pulls out the 8
canonical swing positions:

    address, toe_up, mid_backswing, top, mid_downswing, impact,
    mid_follow_through, finish

Usage:
    python swing_frames.py my_swing.mp4
    python swing_frames.py my_swing.mp4 --compare pro_swing.mp4
    python swing_frames.py my_swing.mp4 --outdir out --model full --no-overlay

Outputs (per video, under --outdir):
    01_address.png ... 08_finish.png   event frames (pose skeleton overlaid)
    contact_sheet.png                  all 8 side by side
    events.json                        frame numbers and timestamps
    comparison.png                     only with --compare: 2 rows, aligned by position

First run downloads the pose model (~9 MB) into models/ next to this script.
Slo-mo video (iPhone 240fps) gives much better odds of a clean impact frame.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

EVENTS = [
    "address", "toe_up", "mid_backswing", "top",
    "mid_downswing", "impact", "mid_follow_through", "finish",
]

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

L_WRIST, R_WRIST = 15, 16

# Body-only skeleton (face landmarks skipped)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
    (27, 31), (28, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]


def get_model(name: str) -> Path:
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / f"pose_landmarker_{name}.task"
    if not path.exists():
        print(f"Downloading pose model '{name}' (one-time, ~9 MB)...")
        urllib.request.urlretrieve(MODEL_URLS[name], path)
    return path


def rotate_frame(frame, rotate: int):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def track_video(video_path: Path, model_path: Path, rotate: int):
    """Run pose on every frame. Returns (frames_bgr, landmarks_per_frame, fps).

    landmarks_per_frame[i] is a list of 33 (x, y) pixel tuples, or None if no
    pose was detected in that frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    frames, landmarks = [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = rotate_frame(frame, rotate)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_img, ts_ms)
            if result.pose_landmarks:
                pts = [(lm.x * w, lm.y * h) for lm in result.pose_landmarks[0]]
            else:
                pts = None
            frames.append(frame)
            landmarks.append(pts)
            idx += 1
            if idx % 100 == 0:
                print(f"  {idx} frames processed...")
    cap.release()
    if not frames:
        sys.exit(f"No frames read from {video_path}")
    return frames, landmarks, fps


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)  # odd, >= 3
    kernel = np.ones(window) / window
    return np.convolve(np.pad(x, window // 2, mode="edge"), kernel, mode="valid")


def wrist_track(landmarks, n_frames: int, fps: float):
    """Wrist-midpoint trajectory with gaps interpolated. Returns (xs, ys)."""
    ys = np.full(n_frames, np.nan)
    xs = np.full(n_frames, np.nan)
    for i, pts in enumerate(landmarks):
        if pts is not None:
            xs[i] = (pts[L_WRIST][0] + pts[R_WRIST][0]) / 2
            ys[i] = (pts[L_WRIST][1] + pts[R_WRIST][1]) / 2
    valid = ~np.isnan(ys)
    if valid.sum() < max(10, n_frames * 0.5):
        sys.exit(
            f"Pose detected in only {int(valid.sum())}/{n_frames} frames. "
            "Check that the golfer is fully in frame (try --rotate 90/180/270 "
            "if the video looks sideways, or --model heavy)."
        )
    idx = np.arange(n_frames)
    ys = np.interp(idx, idx[valid], ys[valid])
    xs = np.interp(idx, idx[valid], xs[valid])
    win = max(3, int(fps / 15))
    ys, xs = smooth(ys, win), smooth(xs, win)
    return xs, ys


def crossing(ys, start, end, level, direction):
    """First frame in [start, end) where ys crosses `level`.

    direction "up" means y decreasing through level (hands rising),
    "down" means y increasing through level.
    """
    for i in range(start, min(end, len(ys))):
        if direction == "up" and ys[i] <= level:
            return i
        if direction == "down" and ys[i] >= level:
            return i
    return None


def detect_events(xs: np.ndarray, ys: np.ndarray, fps: float) -> dict:
    """Map the 8 swing positions to frame indices from wrist position alone.

    Purely position-based (hand speed is never used) so it works on
    continuous footage, slo-mo, and pause-and-step analysis videos alike,
    where speed is zero on freeze frames and spikes at cuts. Structure: the
    hands go high twice (backswing top, then finish hold). Find those two
    episodes, then anchor everything else around and between them.
    """
    n = len(ys)
    baseline = float(np.percentile(ys, 97))  # hands-down (address) height
    rng = baseline - float(np.min(ys))       # positive: high hands = small y
    if rng < 20:
        sys.exit("No backswing motion detected. Is the clip trimmed to one swing?")

    up = ys < baseline - 0.6 * rng
    episodes, start = [], None
    for i, flag in enumerate(up):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, n))
    episodes = [(a, b) for a, b in episodes if b - a >= 3]  # drop noise blips
    if not episodes:
        sys.exit("No backswing motion detected. Is the clip trimmed to one swing?")

    e1a, e1b = episodes[0]
    top = e1a + int(np.argmin(ys[e1a:e1b]))

    if len(episodes) > 1:
        e2a, e2b = episodes[-1]
        finish = e2a + int(np.argmin(ys[e2a:e2b]))
        impact_zone_end = e2a
    else:
        # Clip ends before a distinct finish hold (e.g. cut right after impact)
        finish = n - 1
        impact_zone_end = n
    # Address: if the clip starts with the hands already set (recommended
    # trimming), the first frame is the cleanest address. Otherwise take the
    # last near-baseline frame, backed off by the smoothing window so the
    # takeaway boundary (smeared by smoothing at hard cuts) is not picked.
    addr = [i for i in range(top) if ys[i] >= baseline - 0.05 * rng]
    win = max(3, int(fps / 15))
    if not addr or addr[0] == 0:
        address = 0
    else:
        address = max(addr[-1] - win, addr[0])

    # Impact: hands at their lowest between the top and the finish episode.
    # At regular framerates this lands 1-2 frames after the strike (VIDEO
    # mode's temporal prior lags the motion-blurred hands through the
    # hitting zone); refine_impact() re-measures the window afterwards.
    impact = top + int(np.argmax(ys[top:impact_zone_end]))

    toe_up = crossing(ys, address, top, baseline - 0.30 * rng, "up")
    mid_back = crossing(ys, address, top, baseline - 0.70 * rng, "up")
    mid_down = crossing(ys, top, impact, baseline - 0.50 * rng, "down")
    mid_follow = crossing(ys, impact + 1, finish + 1, baseline - 0.50 * rng, "up")

    events = {
        "address": address, "toe_up": toe_up, "mid_backswing": mid_back,
        "top": top, "mid_downswing": mid_down, "impact": impact,
        "mid_follow_through": mid_follow, "finish": finish,
    }
    # Fall back to neighbor midpoints for any crossing that wasn't found
    order = [events[e] for e in EVENTS]
    for i, f in enumerate(order):
        if f is None:
            prev = next(order[j] for j in range(i - 1, -1, -1) if order[j] is not None)
            nxt = next((order[j] for j in range(i + 1, len(order)) if order[j] is not None), n - 1)
            order[i] = (prev + nxt) // 2
    return dict(zip(EVENTS, order))


def refine_impact(frames, model_path, lo, hi, fallback):
    """Re-measure wrist height over [lo, hi) with IMAGE-mode pose and return
    the hands-lowest frame.

    IMAGE mode detects each frame independently, so unlike the VIDEO-mode
    track it does not lag the motion-blurred hands through the hitting zone.
    Too slow to run on every frame; only worth it on this small window.
    """
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
    )
    measured = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(lo, hi):
            h = frames[i].shape[0]
            rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.pose_landmarks:
                pts = result.pose_landmarks[0]
                y = (pts[L_WRIST].y + pts[R_WRIST].y) / 2 * h
                measured.append((i, y))
    if not measured:
        return fallback
    # Ball contact comes just after the hands' lowest point (teed ball, hit
    # on the upswing), and consecutive frames near the bottom measure within
    # noise of each other: take the last frame within a hair of the lowest.
    bottom = max(y for _, y in measured)
    eps = 0.05 * (bottom - min(y for _, y in measured))
    return max(i for i, y in measured if y >= bottom - eps)


def joint_angles(pts):
    """2D angles in degrees at the major joints, plus spine tilt from
    vertical. Angles are size- and distance-invariant, so two bodies at the
    same camera angle can be compared directly."""
    def ang(a, b, c):
        v1 = np.array(pts[a]) - np.array(pts[b])
        v2 = np.array(pts[c]) - np.array(pts[b])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        # Coincident landmarks have no defined angle; 90 matches what the old
        # epsilon denominator produced. An epsilon in the normal path sits
        # where acos is steepest, so a straight limb measured 179.99991 and
        # scale invariance broke at 1e-4 degrees. Matches angles.ts.
        if n1 == 0 or n2 == 0:
            return 90.0
        cos = np.dot(v1, v2) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    angles = {
        "l_elbow": ang(11, 13, 15), "r_elbow": ang(12, 14, 16),
        "l_shoulder": ang(13, 11, 23), "r_shoulder": ang(14, 12, 24),
        "l_hip": ang(11, 23, 25), "r_hip": ang(12, 24, 26),
        "l_knee": ang(23, 25, 27), "r_knee": ang(24, 26, 28),
    }
    sh = (np.array(pts[11]) + np.array(pts[12])) / 2
    hp = (np.array(pts[23]) + np.array(pts[24])) / 2
    v = sh - hp
    angles["spine_tilt"] = float(np.degrees(np.arctan2(v[0], -v[1])))
    return angles


def compare_swings(marks_a, events_a, marks_b, events_b):
    """Joint-angle similarity at each matched position.

    Per position: mean absolute angle difference across the 9 measured
    angles, mapped to a 0-100 score (100 = identical, 2 points lost per
    degree of average difference). Only meaningful when both videos are
    shot from the same camera angle and the golfers share handedness.
    """
    result = {}
    for name in EVENTS:
        pa, pb = marks_a[events_a[name]], marks_b[events_b[name]]
        if pa is None or pb is None:
            result[name] = None
            continue
        aa, ab = joint_angles(pa), joint_angles(pb)
        diffs = {k: abs(aa[k] - ab[k]) for k in aa}
        mean_diff = sum(diffs.values()) / len(diffs)
        result[name] = {
            "score": round(max(0.0, 100 - 2 * mean_diff), 1),
            "mean_angle_diff_deg": round(mean_diff, 1),
            "joint_diffs_deg": {k: round(v, 1) for k, v in diffs.items()},
        }
    return result


def draw_pose(frame, pts):
    out = frame.copy()
    if pts is None:
        return out
    ipts = [(int(x), int(y)) for x, y in pts]
    for a, b in POSE_CONNECTIONS:
        cv2.line(out, ipts[a], ipts[b], (0, 255, 140), 2, cv2.LINE_AA)
    for x, y in ipts[11:]:
        cv2.circle(out, (x, y), 4, (0, 140, 255), -1, cv2.LINE_AA)
    return out


def label(img, text, height=36):
    h, w = img.shape[:2]
    bar = np.zeros((height, w, 3), dtype=np.uint8)
    scale = 0.7
    while scale > 0.4 and cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 12:
        scale -= 0.05
    cv2.putText(bar, text, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def tile_resize(img, tile_h=360):
    scale = tile_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), tile_h))


def pad_to_width(img, width):
    left = (width - img.shape[1]) // 2
    right = width - img.shape[1] - left
    return cv2.copyMakeBorder(img, 0, 0, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def contact_sheet(tiles):
    return cv2.hconcat([label(tiles[n], n.replace("_", " ")) for n in EVENTS])


def comparison_sheet(tiles_a, tiles_b, similarity=None):
    """One column per position: both videos' tiles padded to a shared width
    so every position lines up regardless of the videos' aspect ratios.
    With similarity, each column label gets its 0-100 score appended."""
    cols = []
    for name in EVENTS:
        a, b = tiles_a[name], tiles_b[name]
        w = max(a.shape[1], b.shape[1])
        col = cv2.vconcat([pad_to_width(a, w), pad_to_width(b, w)])
        text = name.replace("_", " ")
        if similarity and similarity.get(name):
            text += f"  {similarity[name]['score']:.0f}"
        cols.append(label(col, text))
    return cv2.hconcat(cols)


def process(video_path: Path, model_path: Path, outdir: Path, rotate: int, overlay: bool):
    print(f"\nProcessing {video_path.name}...")
    frames, landmarks, fps = track_video(video_path, model_path, rotate)
    print(f"  {len(frames)} frames at {fps:.1f} fps")
    xs, ys = wrist_track(landmarks, len(frames), fps)
    events = detect_events(xs, ys, fps)

    # Second look at impact with lag-free per-frame detection, searching
    # mostly backwards since the tracked impact is only ever late
    win = max(3, int(fps / 15))
    lo = max(events["top"], events["impact"] - 2 * win)
    hi = min(len(frames), events["impact"] + win + 1)
    events["impact"] = refine_impact(frames, model_path, lo, hi, events["impact"])

    outdir.mkdir(parents=True, exist_ok=True)
    tiles = {}
    for i, name in enumerate(EVENTS, 1):
        f = events[name]
        img = draw_pose(frames[f], landmarks[f]) if overlay else frames[f]
        tiles[name] = tile_resize(img)
        cv2.imwrite(str(outdir / f"{i:02d}_{name}.png"), img)

    cv2.imwrite(str(outdir / "contact_sheet.png"), contact_sheet(tiles))
    with open(outdir / "events.json", "w") as fh:
        json.dump({
            "video": str(video_path), "fps": fps, "n_frames": len(frames),
            "events": {k: {"frame": int(v), "time_s": round(v / fps, 3)}
                       for k, v in events.items()},
        }, fh, indent=2)

    print(f"  Events: " + ", ".join(f"{k}={v}" for k, v in events.items()))
    print(f"  Wrote {outdir}\\contact_sheet.png")
    return tiles, landmarks, events


def main():
    ap = argparse.ArgumentParser(description="Extract golf swing position frames.")
    ap.add_argument("video", type=Path, help="video of a single swing")
    ap.add_argument("--compare", type=Path, metavar="PRO_VIDEO",
                    help="second video (e.g. pro swing) for side-by-side comparison")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="output dir (default: out/<video name> next to this script)")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                    help="rotate frames clockwise if the video reads sideways")
    ap.add_argument("--no-overlay", action="store_true", help="skip pose skeleton overlay")
    args = ap.parse_args()

    model_path = get_model(args.model)
    root = args.outdir or (Path(__file__).parent / "out")
    overlay = not args.no_overlay

    tiles, marks, events = process(args.video, model_path, root / args.video.stem,
                                   args.rotate, overlay)
    if args.compare:
        pro_tiles, pro_marks, pro_events = process(
            args.compare, model_path, root / args.compare.stem, args.rotate, overlay)

        similarity = compare_swings(marks, events, pro_marks, pro_events)
        scored = {k: v for k, v in similarity.items() if v}
        overall = round(sum(v["score"] for v in scored.values()) / len(scored), 1) \
            if scored else None

        print("\nSimilarity by position (100 = identical joint angles):")
        for name in EVENTS:
            v = similarity[name]
            if v is None:
                print(f"  {name:20s}   n/a (no pose at this frame)")
                continue
            gaps = sorted(v["joint_diffs_deg"].items(), key=lambda kv: -kv[1])[:2]
            gap_txt = ", ".join(f"{k} {d:.0f}" for k, d in gaps)
            print(f"  {name:20s} {v['score']:5.1f}   biggest gaps (deg): {gap_txt}")
        print(f"  {'overall':20s} {overall}")

        stem = f"{args.video.stem}_vs_{args.compare.stem}"
        with open(root / f"similarity_{stem}.json", "w") as fh:
            json.dump({"video": str(args.video), "compare": str(args.compare),
                       "overall_score": overall, "positions": similarity}, fh, indent=2)

        combo = comparison_sheet(tiles, pro_tiles, similarity)
        combo_path = root / f"comparison_{stem}.png"
        cv2.imwrite(str(combo_path), combo)
        print(f"\nSide-by-side comparison: {combo_path}")


if __name__ == "__main__":
    main()
