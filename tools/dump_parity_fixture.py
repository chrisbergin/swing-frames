"""Dump reference output from the Python pipeline as a test fixture.

The web port re-implements wrist tracking and event detection in TypeScript.
This captures what the Python actually produces on a real clip so the port can
be tested against it directly, rather than only against synthetic curves.

Everything here calls into swing_frames.py, so the fixture is real reference
output and regenerating it after a Python change will surface any drift.

Usage:
    python tools/dump_parity_fixture.py IMG_5146.MOV Grant-Horvat-Driver2.mp4

Writes web/src/core/__fixtures__/<video stem>.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from swing_frames import (  # noqa: E402
    L_WRIST,
    R_WRIST,
    detect_events,
    get_model,
    track_video,
    wrist_track,
)

OUTDIR = ROOT / "web" / "src" / "core" / "__fixtures__"


def dump(video: Path, model_path: Path) -> None:
    frames, landmarks, fps = track_video(video, model_path, rotate=0)
    n = len(frames)

    # Raw per-frame wrist midpoints, before interpolation and smoothing. This
    # is the pipeline's real input, so the port can be fed exactly the same
    # numbers without needing to run pose detection.
    wrists = []
    for pts in landmarks:
        if pts is None:
            wrists.append(None)
        else:
            wrists.append([
                round((pts[L_WRIST][0] + pts[R_WRIST][0]) / 2, 6),
                round((pts[L_WRIST][1] + pts[R_WRIST][1]) / 2, 6),
            ])

    xs, ys = wrist_track(landmarks, n, fps)
    # detect_events runs before refine_impact, which needs decoded frames and
    # image-mode pose, so impact here is the tracked value, not the refined one.
    events = detect_events(xs, ys, fps)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{video.stem}.json"
    with open(out, "w") as fh:
        json.dump({
            "video": video.name,
            "fps": fps,
            "n_frames": n,
            "detected_frames": sum(1 for w in wrists if w is not None),
            "wrists": wrists,
            "smoothed_ys": [round(float(v), 6) for v in ys],
            "events": {k: int(v) for k, v in events.items()},
        }, fh)
    print(f"  events: {events}")
    print(f"  wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model_path = get_model("full")
    for arg in sys.argv[1:]:
        video = (ROOT / arg) if not Path(arg).is_absolute() else Path(arg)
        if not video.exists():
            sys.exit(f"No such video: {video}")
        dump(video, model_path)


if __name__ == "__main__":
    main()
