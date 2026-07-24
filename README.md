# swing-frames

Takes a video of a single golf swing and automatically extracts a still frame for each of the 8 canonical swing positions: address, toe up, mid backswing, top, mid downswing, impact, mid follow through, finish. Optionally lines up two videos (yours vs a pro) position by position in one comparison image, and scores how similar the two swings are at each position.

## Usage

```
pip install -r requirements.txt
python swing_frames.py my_swing.mp4
python swing_frames.py my_swing.mp4 --compare pro_swing.mp4
```

Options: `--model lite|full|heavy` (default full), `--rotate 90|180|270` for sideways videos, `--no-overlay` to skip the skeleton, `--outdir`. Outputs land in `out/<video name>/`: 8 labeled PNGs, `contact_sheet.png`, `events.json`, and in compare mode `comparison_*.png` plus `similarity_*.json`.

Recording guidance:

- Trim the clip to a single swing (walking around or practice swings before the swing will confuse the detection)
- Golfer fully in frame the whole swing
- Slo-mo (iPhone 240fps) strongly recommended, otherwise impact may land between frames
- Down-the-line or face-on both work, but compare like angle with like angle
- Pro clips from YouTube must be trimmed to one continuous swing from one camera

## Technology

- **MediaPipe Pose (PoseLandmarker, Tasks API)**: Google's pretrained human pose model. Given an image, it returns 33 body keypoints (shoulders, elbows, wrists, hips, knees, ankles, plus face and hand points) as x/y coordinates. This is the only "AI" in the system: pretrained, runs locally on CPU, no training and no cloud calls.
- **OpenCV (cv2)**: video decoding, image drawing, resizing, and writing the output PNGs.
- **NumPy**: the trajectory math (smoothing, percentiles, argmax/argmin over the wrist track).

Deliberate constraints: body keypoints only (the club and ball are never tracked), and no learned event model. All 8 positions are found with geometric heuristics on one signal: where the wrists are. If the heuristics ever prove too brittle, the documented fallback is SwingNet (McNally et al. 2019), an ML model trained specifically to label swing events.

## System Requirements

- Python 3.13
- `pip install -r requirements.txt` (mediapipe, opencv-python, numpy; mediapipe 0.10.35 verified)
- ~9 MB pose model, auto-downloaded to `models/` on first run (one-time, needs internet)
- CPU is fine, no GPU needed. A 5-second clip processes in well under a minute; long clips scale linearly with frame count.
- Input: any video OpenCV can read (iPhone .MOV and .mp4 both verified). Trim the clip to a single swing. Slo-mo strongly recommended for a crisp impact frame.

## How It Works, Step by Step

1. **Track**: run MediaPipe Pose over every frame (VIDEO mode, which uses the previous frame as a prior for speed and stability). Keep the wrist midpoint, the average of the left and right wrist positions, as one (x, y) point per frame.
2. **Clean the track**: frames where no pose was found are filled by linear interpolation, then the track is smoothed with a small moving average (window scales with fps). The result is `ys` (wrist height per frame) and `xs`.
3. **Find the two hands-high episodes**: in every swing the hands go high exactly twice, at the top of the backswing and in the finish hold. The detector computes a baseline (address-height, 97th percentile of `ys`) and marks every run of frames where the hands are well above it. First episode's peak = **top**. Last episode's peak = **finish**.
4. **Anchor the rest around them**:
   - **address**: frame 0 if the clip starts with hands already set (the normal case), otherwise the last near-baseline frame before the takeaway, backed off a few frames so boundary smear is never picked
   - **impact**: hands at their lowest point between top and the finish episode, then refined (step 5)
   - **toe up, mid backswing, mid downswing, mid follow through**: the frames where wrist height crosses fixed fractions of the swing's height range (30%, 70%, 50%, 50%) between the neighboring anchors
   - any crossing that is not found falls back to the midpoint of its neighbors
5. **Refine impact**: the tracked hands-lowest frame can be one frame off the true strike, because a driver contacts the teed ball just after the hands' low point and neighboring frames measure within noise of each other. A second, small pose pass (IMAGE mode, each frame detected independently) re-measures a bounded window around the candidate and takes the last frame within 5% of the lowest hands.
6. **Output**: for each event, write a PNG with the skeleton overlaid, plus `contact_sheet.png` (all 8 side by side), `events.json` (frame numbers and timestamps), and in `--compare` mode a two-row comparison image with both videos aligned column by column per position.
7. **Similarity scoring** (`--compare` mode): at each matched position, both poses are reduced to 9 joint angles (both elbows, shoulders, hips, knees, plus spine tilt from vertical). Angles are size- and distance-invariant, so the two bodies compare directly. Per position: mean absolute angle difference mapped to a 0-100 score (2 points lost per degree of average difference). Output: a console table with each position's score and its two biggest joint gaps, an overall score (mean across positions), `similarity_*.json` with every angle difference, and the score stamped on each comparison sheet column. Caveats: both videos must be shot from the same camera angle and both golfers must share handedness, and a low score can mean a technique difference or an event-timing difference (the extracted frames sitting at slightly different moments of the swing).

Why position-based instead of speed-based: hand speed is zero on freeze frames and spikes at cuts, so speed-based detection broke on pause-and-step analysis videos (the kind golf YouTubers post). Wrist position works on continuous footage, slo-mo, and step-frame edits alike. Two earlier speed-based versions were scrapped over this.

## Code Map (`swing_frames.py`)

Single file, ~370 lines, top to bottom:

- `EVENTS`, `MODEL_URLS`, `POSE_CONNECTIONS`: the 8 position names, model download URLs, and which keypoint pairs form the drawn skeleton
- `get_model()`: downloads/caches the pose model file
- `rotate_frame()`: applies `--rotate` for sideways videos
- `track_video()`: opens the video, runs the pose model on every frame, returns raw frames + per-frame landmarks + fps
- `smooth()`, `wrist_track()`: build the clean wrist-midpoint trajectory (`xs`, `ys`) from raw landmarks
- `crossing()`: helper that finds where the track crosses a given height level in a given direction
- `detect_events()`: the heuristic core described above; maps the 8 positions to frame indices using only `xs`/`ys`
- `refine_impact()`: the bounded IMAGE-mode second pass for the impact frame
- `joint_angles()`, `compare_swings()`: reduce a pose to 9 joint angles and score two swings' similarity per position
- `draw_pose()`, `label()`, `tile_resize()`, `pad_to_width()`: drawing and layout helpers
- `contact_sheet()`, `comparison_sheet()`: assemble the multi-position output images
- `process()`: per-video pipeline (track, detect, refine, write outputs)
- `main()`: CLI argument handling; runs `process()` on one or two videos and writes the comparison

## Gotchas Learned the Hard Way

- **Never extract reference frames with `cap.set(CAP_PROP_POS_FRAMES, n)`**: OpenCV seeking is off by 2-3 frames on iPhone HEVC video. Always read sequentially when checking what frame N actually shows.
- Pro clips from YouTube must be trimmed to one continuous swing from one camera; replays and multi-angle edits confuse the episode detection.
- Regular-framerate video (27-30fps) works, but the strike lands between frames; slo-mo makes the impact frame exact.
