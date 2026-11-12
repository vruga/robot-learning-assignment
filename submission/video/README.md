# Demo video: `cr_all_demo.mp4` (45 s)

What the assignment asks for: *a short video showing multiple initial slots and at least one failure/recovery when
applicable.* This is that video. It is built from the evaluator's own recorder (`--video-episodes`), so the frames are
exactly what the policy sees.

## What is in it

A real evaluator run of `cr_all` (the submitted policy), started at seed 7,310,003 so that it begins with the
known miss: **29 of 30 successes, a valid qualifying run** (30 trials, 5 per source slot, integrity ok, submission
unchanged). The first six episodes were recorded; four are shown (2x speed, three camera views side by side):

| Clip | Seed | Start slot | Outcome |
|---|---|---|---|
| 1 | 7,310,004 | 6 | success in 13.8 s |
| 2 | 7,310,005 | 1 | success in 15.8 s |
| 3 | 7,310,007 | 4 | success in 14.6 s |
| 4 | 7,310,003 | 4 | **failure**: timeout at 34 s |

Views, left to right: head, right wrist, left wrist (84x84 each). They are the policy's low-resolution inputs, not a
third-person camera. The captions are generated from `report.json`, not typed by hand.

## The failure (and the honest answer about "recovery")

The policy has no explicit recovery behaviour, and the failure shown **does not recover**. From the recorded actions of
that episode (seed 7,310,003, slot 4):

- the policy closes the left gripper at about 6 s;
- from about 8 s to the end (34 s) its commanded actions are almost constant: base velocity 0, left-arm command change
  0.0002 per step (versus 0.002-0.007 per step while it is moving), gripper command stays closed;
- the successful episodes keep going at that point: for example seed 7,310,004 starts its sideways walk at about 8 s and
  releases the plate by about 12 s.

So this is a **stall**: the policy reaches a state where it repeats the same command forever. I have not verified from the
data whether the plate was actually held when it froze. This is why the natural next fix is an observation-only stall
detector that triggers a retry; it has not been built.

## Reproduce

```bash
# 1. record: the evaluator writes episode_000.mp4 ... for the first N episodes of a run
BC_CHECKPOINT=<cr_all checkpoint> BC_EXECUTE=20 uv run --extra video advaitic-eval \
  --policy-root <dir with submission/ and the checkpoint> --policy submission.bc_policy:make_policy \
  --policy-python <torch env> --episodes 30 --seed-start 7310003 --video-episodes 6 --out results/video_cr_all
# 2. assemble
python submission/scripts/make_video.py --clips results/video_cr_all --episodes 1 2 4 0 \
  --ffmpeg <ffmpeg> --font <ttf> --out submission/video/cr_all_demo.mp4
```

(`uv sync --locked --extra video` installs the recorder dependency. `*.mp4` is git-ignored, so the video is added with
`git add -f`.)
