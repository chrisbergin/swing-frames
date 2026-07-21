# Golf Swing Sequencer

## What This Is

Prototype for the golf swing app idea (see `context/ideas.md`): upload a swing video, auto-extract frames at the 8 key swing positions, and optionally compare side by side against a pro's swing. Phase 1 is this local Python prototype. If the frame extraction proves useful on real videos, phase 2 is a client-side web app (React + MediaPipe WASM, free static hosting, video never leaves the browser), reusing the karaoke app stack.

## How It Works

`swing_frames.py` runs MediaPipe Pose (Tasks API, `PoseLandmarker`) over every frame, tracks the wrist midpoint, and detects events from wrist height plus hand speed:

- address: last still frame before sustained motion
- top: highest wrist point before peak hand speed
- impact: wrists return to address height after the top
- toe_up / mid_backswing / mid_downswing / mid_follow_through: fractional height crossings between address and top
- finish: motion settles after impact

No club tracking (body keypoints only) and no ML training. If heuristics prove too brittle, the fallback is SwingNet trained on GolfDB (McNally et al. 2019, code and weights on GitHub, MobileNetV2 so CPU is fine).

## Running It

```
python projects/golf_swing/swing_frames.py my_swing.mp4
python projects/golf_swing/swing_frames.py my_swing.mp4 --compare pro_swing.mp4
```

Options: `--model lite|full|heavy` (default full), `--rotate 90|180|270` for sideways videos, `--no-overlay`, `--outdir`. Outputs land in `out/<video name>/`: 8 labeled PNGs, `contact_sheet.png`, `events.json`, and `comparison_*.png` in compare mode.

Requirements: Python 3.13 system install, `pip install mediapipe opencv-python numpy` (done 2026-07-21, mediapipe 0.10.35). The pose model auto-downloads to `models/` on first run. `models/` and `out/` are gitignored.

## Recording Guidance

- Trim the clip to a single swing (walking around or practice swings before the swing will confuse the motion-start detection)
- Golfer fully in frame the whole swing
- iPhone slo-mo (240fps) strongly recommended, otherwise impact may land between frames
- Down-the-line or face-on both work for extraction, but compare like angle with like angle

## Status / Next Steps

- 2026-07-21: script scaffolded, dependencies installed, pipeline verified end to end on a synthetic clip (pose detection correctly reports no human found). Not yet run on a real swing video.
- [ ] Record 3-4 slo-mo swings (down-the-line) and run extraction; judge frame quality
- [ ] Grab a pro swing video at the same angle and test --compare
- [ ] If heuristics are shaky: tune crossing fractions, or evaluate SwingNet
- [ ] If extraction is good: decide on phase 2 web app
