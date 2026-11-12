# Candidate README: G1 plate transfer

**Shipped policy:** image + proprioception behaviour cloning (ResNet18 on 3 cameras, 20-step action chunks, executed
open loop) distilled from a privileged scripted expert, plus a small observation-only stall breaker.
Weights: `submission/weights/final.pt` (57 MB, sha256 `92f0318cad607e9db7e2a087a8890b524713c018564deaf2b91c5b4541e0c67f`).
`make_policy()` loads them with no environment variables.

## Results (our own `advaitic-eval` runs; not yet independently re-run by a reviewer)

All runs: same weights, same code (one submission digest), 20 actions executed per chunk, balanced source slots.
Every miss was a timeout or a placement drop; **0 falls and 0 policy errors** in all 3,630 trials.

| Run | Seeds | Trials | Success | Wilson 95% |
|---|---|---|---|---|
| Public block (qualifying run, with video) | 7,310,000+ | 30 | **30 (100%)**, needed 27 | – |
| Fresh screening block | 7,600,000-7,630,000 | 600 | **598 (99.7%)** | 98.8-99.9% |
| Fresh certification run | 7,700,000-7,790,000 (10 disjoint blocks) | 3,000 | **2,970 (99.0%)** | 98.6-99.3% |
| Both fresh runs pooled | – | 3,600 | **3,568 (99.1%)** | 98.7-99.4% |

Per source slot over the 3,600 fresh trials (out of 600 each): 599, 600, 600, 595, 593, 581 for slots 1-6.
Raw evaluator reports: `reports/final_v5/`. Seeds in the two fresh runs were never used for training or tuning.
Caveats: the public block was looked at during development (the execute length was chosen there). A 99.9% goal was
set and **not reached**: the best estimate of the true success rate is about 99.0-99.1%. Block-to-block spread is large
(0 to 7 misses per 300 trials in the certification run), which is why the large run is the number to trust.

## How it works

- **Observations / actions.** Three 84x84 RGB cameras (head, right wrist, left wrist) and 50-d proprioception in;
  a 20-d absolute action out (4 base commands, 14 arm joint targets, 2 grippers), predicted as a 20-step chunk
  and executed in full before re-planning. No simulator state, seed or slot is used at inference.
- **Model** (`model.py`, `bc_policy.py`): one ResNet18 shared over the three cameras, concatenated with normalised
  proprioception, MLP head, L1 loss, random-shift augmentation, EMA weights, 150k steps at batch 128.
- **Data.** 60 human demonstrations plus episodes from a **privileged scripted expert** (`expert.py`): a closed-loop
  pick-and-place that reads the simulator's true plate and rack poses and uses IK on the actual hand. It is
  **training-time only**, never the evaluated policy, and replays nothing. Only successful episodes are kept; expert
  seeds are 5,000,000+, disjoint from every evaluation block. The final training mix is 60 human + 710 new expert
  episodes + 200 earlier expert episodes:
  - 314 clean episodes on hard rack configurations (large x offset, slots 3-6);
  - 396 **DART** episodes: brief random noise bursts on the base and left-arm commands while the expert keeps its own
    command integrator, so the stored label is always the clean expert action (this teaches recovery);
  - 200 clean earlier expert episodes (random sample).
- **Stall breaker** (`bc_policy.py`, observations only): a deterministic single-frame policy can sit in a fixed point
  (for example gripper closed but arm frozen after the grasp). If the arm and gripper joints change by less than
  0.02 (normalised) for 3 consecutive replans, noise is added to the proprioceptive input with a randomly drawn
  strength. On 120 healthy dev episodes it fired once (a slot-6 freeze that it rescued; 120/120 succeeded), and it rescued freezes that otherwise ran out the clock
  (illustrated in `media/seed7520101_breaker_off.mp4` vs `seed7520101_breaker_on.mp4`).

## Baseline and improvements (full log: `EXPERIMENTS.md`)

| Step | Result (execute 20) |
|---|---|
| 60 human demos only | 78% (47/60) |
| + 298 expert episodes | 95-98% |
| + 1,329 expert episodes, slots 4-6 oversampled (`cr_all`) | 98.9% (267/270) |
| Diagnose misses; stall breaker; hard-config + DART data, retrain (`cr_v5`, shipped) | 99.0% (2,970/3,000), 99.7% (598/600) |

Execute length matters a lot for a single-frame policy: the same early weights scored 50% / 73% / 90% at execute
5 / 10 / 20. We did not isolate whether the gain of `cr_v5` over `cr_all` comes from the DART data, the
hard-configuration data, more steps, or the stall breaker.

### Discrete vs continuous action heads (arXiv:2610.20429 idea)

We tested the paper's claim that discrete heads beat continuous regression, in three steps:

- A simplified discrete head (MLP, 256 bins, greedy) was much worse: 35% vs 98% at execute 20; classifier-free
  guidance collapsed it (0-2%).
- A faithful version (transformer head, 512 bins, discrete flow, 20 steps) works, but is **not better** at execute 20:
  108/120 (90.0%) vs 116/120 (96.7%) for our continuous L1 head on the same two blocks (Fisher p = 0.067, not
  significant, so "not better" rather than "worse"). It was trained for about 5% of the paper's sample budget.
- **The paper's mechanism does hold here:** discrete flow is much more reactive when it re-plans often. At execute 5
  it scores 48/60 vs 29/60 for the continuous head (p = 0.0005); at execute 10, 50/60 vs 47/60. The continuous head
  is saturated at execute 20, so it does not need that reactivity. Likely reasons the paper's gain does not show up:
  our baseline uses L1 (median) regression rather than squared error, and our data is mostly one deterministic
  expert, so there are few modes to average. Untested ideas: relative-action targets, more flow steps, a full
  sample budget.

### What did not help

DAgger round 1 (labels anchored to the student's own command), 50-step chunks (equal), pi0.5 zero-shot and a
5k-step fine-tune (0/30), constant-noise DART (broke the expert's phase switches), and **a second round of extra
place-phase perturbation data (`cr_v6`: 592/600, worse than `cr_v5` on the same seeds)**. The complete list of runs,
including invalid and negative ones, is in `EXPERIMENTS.md`; raw reports are in `reports/`.

## Failure modes (30 misses in 3,000 fresh trials)

19 are slot 6 (481/500), 9 are slots 4-5. Replaying misses with read-only privileged probes
(`scripts/diagnose_miss.py`) shows four types: (1) freeze right after the grasp (largely fixed by the stall breaker);
(2) hover a few cm from the slot, sometimes ending with the plate dropped; (3) release a few cm off the slot, after
which the plate rests misaligned and the policy idles; (4) a drop at placement time on slot 6 that leaves the plate
lying flat (not recoverable). A probe on six slot-6 hover seeds suggests strong stall-breaker kicks can themselves
open the gripper mid-hold; a gentler kick looked better but was not evaluated and is **not** in the shipped policy.

## Next steps

Evaluate a gentler stall kick on fresh seeds; diagnose where in the place phase slot-6 plates slip; add a
visual or force cue for release timing; test a history-conditioned policy (frames or recent actions) instead of
single-frame input; check data-scale and ablation questions above with repeated training seeds.

## Reproduce

Setup follows `PROBLEM.md` (uv env for the simulator plus a Python 3.11 torch env, `.venv-train`).

```bash
# evaluate the shipped policy (weights are in the repo); first 3 episodes recorded
uv sync --locked --extra video
BC_DEVICE=cpu MUJOCO_GL=egl uv run advaitic-eval --policy submission.bc_policy:make_policy \
  --policy-python .venv-train/bin/python --episodes 30 --video-episodes 3 --out results/final

# training data (seeds never in an evaluation block)
MUJOCO_GL=egl uv run python -m submission.generate_data --start-seed 5003000 --hard 320 --out <dir>   # clean, hard configs
MUJOCO_GL=egl uv run python -m submission.generate_data --start-seed 5004000 --count 700 \
  --dart 0.03,0.08 --out <dir>                                                                         # DART bursts
.venv-train/bin/python -m submission.pack_data --demos datasets/move_plate.npz --expert-dir <staged dir> --out <packed>
.venv-train/bin/python -m submission.train --packed <packed> --head continuous \
  --boost-slots 3,4,5 --boost 2.0 --steps 150000 --out <ckpt dir>
```

For a clean evaluation run from an isolated copy, pass `--policy-root` (the evaluator hashes the policy folder, so
files changing during a run invalidate it).

## Media

`media/public_episode_00{0,1,2}.mp4`: the evaluator's recordings of the first three public-block episodes
(slots 5, 2 and 1), all successes. `media/seed7520101_breaker_off.mp4` / `_on.mp4`: one seed, in-process clip
(not an evaluator run): without the stall breaker the policy freezes and times out; with it, one kick and the task
completes. `video/` holds the older `cr_all` demo, which includes a failure that does not recover.

## Attribution

- BiGym2 environment and task (pinned upstream commit in `src/advaitic_eval/upstream_manifest.json`), the frozen
  GR00T whole-body Balance/Walk controller, the starter evaluator, and MuJoCo: provided by the assignment.
- PyTorch / torchvision (ResNet18, trained from scratch), uv, ruff, pytest.
- LeRobot and `lerobot/pi05_base` (pi0.5): used only for the reference VLA runs (zero-shot and 5k-step fine-tune), not
  in the shipped policy. Discrete-head experiments follow the idea in arXiv:2610.20429 (our own implementation).
- Human demonstrations: the 60 provided with the task.
- AI tooling: Claude Code (Anthropic) was used for coding, running experiments and analysis in this repository.
  The candidate is responsible for understanding and maintaining the solution.
