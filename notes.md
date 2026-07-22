# Golf Swing Sequencer

## What This Is

Prototype for the golf swing app idea (see `context/ideas.md`): upload a swing video, auto-extract frames at the 8 key swing positions, and optionally compare side by side against a pro's swing. Phase 1 is this local Python prototype. If the frame extraction proves useful on real videos, phase 2 is a client-side web app (React + MediaPipe WASM, free static hosting, video never leaves the browser), reusing the karaoke app stack.

## How It Works

`swing_frames.py` runs MediaPipe Pose (Tasks API, `PoseLandmarker`) over every frame, tracks the wrist midpoint, and detects events from wrist height alone. Hand speed is deliberately not used: it is zero on freeze frames and spikes at cuts in pause-and-step analysis videos, which broke two earlier speed-based versions. Structure: hands go high twice per swing (backswing top, finish hold).

- find the two hands-high episodes: top = highest wrists in the first, finish = highest wrists in the last
- impact: hands at their lowest between the top and the finish episode
- address: last frame before the top with hands near their baseline (97th percentile of wrist height)
- toe_up / mid_backswing / mid_downswing / mid_follow_through: fractional height crossings between those anchors

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
- 2026-07-21: first real video (IMG_5146.MOV, down-the-line, driver, 27fps): first run failed because the transition pause at the top read as address stillness; reworked event anchors (top first, then address and impact relative to it). All 8 extracted frames look correct on visual inspection. Regular-speed video works, but impact lands on the nearest frame; slo-mo still preferred for a true impact frame.
- 2026-07-21: first pro comparison (Grant Horvat driver, down-the-line). The raw YouTube clip is a 33s edit with a slo-mo replay, and post-impact events drifted into the replay. Fixed by bounding post-impact search to 2.5s after impact, and by trimming the clip to the first swing (first 100 frames, saved as horvat_driver_trimmed.mp4). Comparison sheet aligns well. Lesson: pro clips must be trimmed to a single continuous swing from one camera, same rule as own videos.
- 2026-07-21: second pro clip (Grant-Horvat-Driver2.mp4) is a pause-and-step analysis video: freeze frames with near-zero motion connected by jumps. Speed-based anchors collapsed (impact/follow/finish all landed on one plateau). Rewrote detection to be purely position-based (two hands-high episodes). All 8 positions now correct on both continuous and step-frame footage; three video types verified (own phone video, trimmed continuous pro clip, step-frame pro clip).
- 2026-07-21: address frame on Driver2 showed the club already in the air. Cause: "last frame near baseline height" lands exactly on the address-to-takeaway cut, and smoothing smears the boundary, so the picked frame is already the takeaway freeze (hands barely move at shaft parallel, only the club does). Fix: address defaults to frame 0 when the clip starts with hands already set (both test videos do); otherwise last near-baseline frame backed off by the smoothing window. Verified on both videos: club on the ground in both address tiles.
- 2026-07-21: impact lands 2 frames (~70ms) late on the 27fps video: frame 62 shows the ball already flying, true impact is frame 60. Diagnosed with raw landmark dumps: MediaPipe lags the motion-blurred hands through the hitting zone and only catches up after impact (tracked hands-lowest at 62-63; physically a driver's hands bottom out before a teed ball, so the lag is tracker error, not real motion). Tried anchoring impact to "hands closest to their address spot": does not discriminate, the lagged track is closest to address at 62-63 too. Wrist data cannot see the club. Options if regular-fps impact matters: ball-patch step detection (a patch around the ball permanently changes at impact; signal confirmed present but weak against handheld camera noise, and needs auto-locating the ball) or SwingNet. Slo-mo footage avoids the blur entirely and is already the recommended format.
- [ ] Record more swings (slo-mo, face-on too) and confirm the heuristics hold up
- [ ] Grab a pro swing video at the same angle and test --compare
- [ ] If heuristics are shaky: tune crossing fractions, or evaluate SwingNet
- [ ] If extraction is good: decide on phase 2 web app
