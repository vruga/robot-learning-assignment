# Problem statement (short version)

**Goal:** A Unitree G1 humanoid picks a plate off rack 1 and places it in any slot of rack 2, then lets go.

**Score:** at least 90% success over 30 trials (27/30), with 5 trials from each of the 6 source slots. Falls, timeouts and policy errors count as failures.

**Success means all of these hold for 1 simulated second:**
- plate is within 5 cm of a rack-2 slot, tilted no more than 20°, and touching rack 2
- both grippers have let go
- plate is not resting on the table, never touched the floor, and the robot never fell

**What the policy sees:** three 84x84 RGB cameras (head, right wrist, left wrist) and 50 proprioception numbers. Nothing else: no seed, slot, plate pose or sim state.

**What the policy outputs:** 20 values in [-1, 1] per step at 50 Hz. These are 4 base commands (forward, lateral, pelvis height, yaw), 14 arm joint targets and 2 grippers. The balance/walk controller is frozen. Chunks of up to 50 actions are allowed.

**Variation:** the plate starts in one of 6 slots, and each rack is shifted by up to ±5 cm in x/y.

**Episode budget:** 1,700 policy steps (34 simulated seconds).

**Allowed:** classical control, state machines, imitation learning, RL, VLAs, perception models. Privileged state is allowed only for disclosed training, never at evaluation.
**Not allowed:** open-loop replay, per-seed lookup, changing the evaluator, physics, controller or success checks.

**Changes go in `submission/` only.**

## Approach: privileged expert, then distill to an image policy

60 human demos (10 per source slot) are too few for 90%, so we generate more data.

1. **Expert** (`submission/expert.py`): a closed-loop scripted pick-and-place that reads the
   simulator's true plate/rack poses. It is **training-time only** and is never the evaluated
   policy. Phases: raise arm, walk to standoff, settle, approach the plate's near rim, pinch,
   lift, strafe, stop and let the gait sway damp, align by arm feedback on the plate pose, lower,
   release, retreat. It uses IK on the actual hand pose, so nothing is replayed.
2. **Data** (`submission/generate_data.py`): run the expert on many seeds, keep only successful
   episodes. Seeds must stay out of the evaluation blocks (the script refuses 7,310,000+).
3. **Student** (`model.py`, `train.py`): behaviour-clone the 3 cameras + proprioception to
   20-step action chunks. The evaluated policy sees only `rgb` and `state`.
4. **DAgger** (next): roll out the student, relabel its visited states with the expert, retrain.
   This teaches recovery, which expert-only data lacks.

Disclose the privileged expert in the report. It is a data-generation workflow, which the
rules allow, and it is never used at inference.

## Would a VLA do better?

Possibly as an ablation, not as the first line.
- **Where it could help:** pretrained visual features generalise better to rack and plate
  position shifts than a ResNet trained from scratch. Candidates: **GR00T N1.x** (humanoid
  focus), **pi0 / pi0.5** (openpi, LoRA fits ~24 GB), **SmolVLA** (small, easy on a 4090).
- **Why it is not the first line:** the task has no language variation, the images are only
  84x84 (VLAs expect 224+; we'd upsample), and the action space is a custom 20-d absolute
  joint/base layout, so every VLA needs a new action head and fine-tuning anyway. The
  bottleneck is closed-loop precision and data quantity, which the expert data addresses.
- **Plan:** get the ResNet student working first. Then fine-tune SmolVLA or GR00T on the same
  expert dataset and compare success per slot. The same data format works for all of them.

## Train on the 4090 (Linux)

Run the commands from the repo root. Don't train on the Mac.

```bash
# 1. Simulator env (headless) + human demos exported to one file
uv sync --locked
uv run bigym-download --task move_plate
MUJOCO_GL=egl uv run python -m submission.export_demos --out datasets/move_plate.npz

# 2. Torch env that ALSO has the simulator (needed for DAgger; training only needs torch)
uv venv --python 3.11 .venv-train
uv pip install --python .venv-train/bin/python -r submission/requirements-train.txt
uv export --frozen --no-hashes --no-emit-project > /tmp/req_locked.txt
uv pip install --python .venv-train/bin/python -r /tmp/req_locked.txt

# 3. Expert data: seeds 5,000,000-5,000,639 (~99% succeed; only successes are kept; ~22 MB/episode)
MUJOCO_GL=egl uv run python -m submission.generate_data \
  --start-seed 5000000 --count 640 --workers 12 --out datasets/expert_train

# 4. Train (images stay in host RAM, ~50 MB/episode: 300 episodes + demos is ~18 GB)
.venv-train/bin/python -m submission.train --data datasets/move_plate.npz \
  --expert-dir datasets/expert_train --max-expert 298 --out checkpoints/bc_expert --steps 60000

# 5. Evaluate (30 trials = public seeds; use --seed-start 7330000 --episodes 60 for a fresh block)
BC_CHECKPOINT=checkpoints/bc_expert/last.pt MUJOCO_GL=egl uv run advaitic-eval \
  --policy submission.bc_policy:make_policy --policy-python $PWD/.venv-train/bin/python \
  --method bc-resnet18-human60+expert298 --checkpoint bc_expert/last.pt --episodes 30 --out results/bc-30
```

Notes
- `BC_EXECUTE` (default **20**, clamped to the model's chunk length) sets how many chunk actions run
  before re-planning. It matters a lot: 5 -> 50%, 10 -> 73%, 20 -> 90% on the same weights.
- Run evaluations from an isolated snapshot (`--policy-root <dir with submission/ + checkpoint>`):
  the evaluator hashes `checkpoints/*.pt` under the root, so a training job writing there invalidates a run.
- The reported model used the 298 episodes from seeds 5,000,000-5,000,639 produced by earlier expert
  versions (see EXPERIMENTS.md section B); regenerating with the final expert gives equivalent data, not
  byte-identical episodes.
- Baseline for comparison: `submission.policy:make_policy` (holds the pose, 0%).
- Models: `vruga/g1-plate-bc-expert` and `vruga/g1-plate-bc-human` on Hugging Face.

All results and what did not work (DAgger round 1, pi0.5 zero-shot): `EXPERIMENTS.md`.

## Mac

Leave `MUJOCO_GL` unset, because `egl` is Linux-only and gives
`RuntimeError: invalid value for environment variable MUJOCO_GL: egl`.
Use the Mac only for `uv run advaitic-smoke` and debugging.
