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
    """Wrist-midpoint trajectory with gaps interpolated. Returns (ys, speeds)."""
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
    speeds = np.hypot(np.gradient(xs), np.gradient(ys))
    speeds = smooth(speeds, win)
    return ys, speeds


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


def detect_events(ys: np.ndarray, speeds: np.ndarray, fps: float) -> dict:
    """Map the 8 swing positions to frame indices using wrist height + speed.

    Anchors are chosen so the transition pause at the top of the backswing
    (near-zero hand speed) can't be confused with address stillness: top is
    the highest wrist point before peak hand speed, and address is the last
    frame before the top where the hands sit near their low baseline.
    """
    n = len(ys)
    peak = int(np.argmax(speeds))          # near impact: fastest hand speed
    still = 0.05 * speeds[peak]
    hold = max(2, int(fps * 0.1))

    top = int(np.argmin(ys[:peak + 1]))
    base_y = float(np.max(ys[:top + 1])) if top > 0 else float(ys[0])
    top_y = ys[top]
    rng = base_y - top_y  # positive: top is higher on screen
    if top == 0 or rng < 20:
        sys.exit("No backswing motion detected. Is the clip trimmed to one swing?")

    addr = [i for i in range(top) if ys[i] >= base_y - 0.05 * rng]
    address = addr[-1] if addr else 0

    toe_up = crossing(ys, address, top, base_y - 0.30 * rng, "up")
    mid_back = crossing(ys, address, top, base_y - 0.70 * rng, "up")

    # Impact: hands' first local bottom after the top, at least halfway back
    # down. (At 30fps the true impact instant can fall between frames.)
    impact = None
    for i in range(top + 2, n - 1):
        if ys[i] >= ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] >= base_y - 0.5 * rng:
            impact = i
            break
    if impact is None:
        impact = min(peak + 1, n - 1)
    mid_down = crossing(ys, top, impact, base_y - 0.50 * rng, "down")

    # Follow-through and finish: search only a bounded window after impact so
    # untrimmed footage (slo-mo replays, camera cuts) can't be grabbed
    horizon = min(n, impact + int(fps * 2.5))
    mid_follow = crossing(ys, impact + 1, horizon, base_y - 0.50 * rng, "up")

    finish = horizon - 1
    calm = speeds < 1.5 * still
    for i in range(min(impact + int(fps * 0.3), horizon - hold), horizon - hold):
        if calm[i:i + hold].all():
            finish = i
            break

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
    cv2.putText(bar, text, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def contact_sheet(frames_by_event, tile_h=360):
    tiles = []
    for name in EVENTS:
        img = frames_by_event[name]
        scale = tile_h / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale), tile_h))
        tiles.append(label(img, name.replace("_", " ")))
    return cv2.hconcat(tiles)


def process(video_path: Path, model_path: Path, outdir: Path, rotate: int, overlay: bool):
    print(f"\nProcessing {video_path.name}...")
    frames, landmarks, fps = track_video(video_path, model_path, rotate)
    print(f"  {len(frames)} frames at {fps:.1f} fps")
    ys, speeds = wrist_track(landmarks, len(frames), fps)
    events = detect_events(ys, speeds, fps)

    outdir.mkdir(parents=True, exist_ok=True)
    frames_by_event = {}
    for i, name in enumerate(EVENTS, 1):
        f = events[name]
        img = draw_pose(frames[f], landmarks[f]) if overlay else frames[f]
        frames_by_event[name] = img
        cv2.imwrite(str(outdir / f"{i:02d}_{name}.png"), img)

    sheet = contact_sheet(frames_by_event)
    cv2.imwrite(str(outdir / "contact_sheet.png"), sheet)
    with open(outdir / "events.json", "w") as fh:
        json.dump({
            "video": str(video_path), "fps": fps, "n_frames": len(frames),
            "events": {k: {"frame": int(v), "time_s": round(v / fps, 3)}
                       for k, v in events.items()},
        }, fh, indent=2)

    print(f"  Events: " + ", ".join(f"{k}={v}" for k, v in events.items()))
    print(f"  Wrote {outdir}\\contact_sheet.png")
    return sheet


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

    sheet = process(args.video, model_path, root / args.video.stem, args.rotate, overlay)
    if args.compare:
        pro_sheet = process(args.compare, model_path, root / args.compare.stem,
                            args.rotate, overlay)
        w = max(sheet.shape[1], pro_sheet.shape[1])
        pad = lambda s: cv2.copyMakeBorder(s, 0, 0, 0, w - s.shape[1],
                                           cv2.BORDER_CONSTANT, value=(0, 0, 0))
        combo = cv2.vconcat([pad(sheet), pad(pro_sheet)])
        combo_path = root / f"comparison_{args.video.stem}_vs_{args.compare.stem}.png"
        cv2.imwrite(str(combo_path), combo)
        print(f"\nSide-by-side comparison: {combo_path}")


if __name__ == "__main__":
    main()
