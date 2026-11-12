# Advaitic · Robot Learning Assignment

Transfer one plate between two racks using a Unitree G1 in MuJoCo. Build a repeatable, observation-driven system, evaluate it honestly, and explain how you improved it.

## Task

Move the plate from rack 1 into **any of the six slots in rack 2**. The source slot varies, and each rack has up to ±5 cm of x/y position variation. The embodiment is the **Unitree G1 with Dex1-1 grippers**, supported by BiGym2's frozen Balance/Walk controller. You do not need to train locomotion.

Build a repeatable closed-loop system. You may use existing VLAs, fine-tuning, imitation learning, RL, perception foundation models, classical planning/control, or combinations. We do not prescribe a particular architecture. AI coding tools are allowed, but you must understand, explain and maintain the solution.

**Performance criterion: at least 90% success across at least 30 trials, with equal coverage of the six source slots.** Valid run sizes are 30, 36, 42, 48, 60, etc. A 30-trial run requires **27 successes**, with five trials from each source slot. Policy errors, falls and timeouts count as failures. Short smoke runs never count as a pass.

The plate must be correctly oriented, near a destination slot, touching the destination rack, released by both grippers, and not supported by the table. These conditions must persist for one simulated second. Touching the floor fails the task; any robot fall prevents a successful episode. We retain BiGym2's checks and episode budget, not a position-only shortcut.

This is the **Advaitic balanced-slot assignment protocol**, not the upstream BiGym2 leaderboard protocol. It changes seed selection and the aggregate pass criterion, while retaining the task physics and success semantics.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone your fork, then:

```bash
uv sync --locked
uv run advaitic-check
```

On headless Linux set `MUJOCO_GL=egl`. On macOS leave it unset. Optional ONNX Runtime telemetry is disabled before import by this package. Downloading pinned dependencies and demonstrations needs internet; no hosted policy service is required.

Verify the real simulator across all six source slots:

```bash
MUJOCO_GL=egl uv run advaitic-smoke
```

Run the interface example through the actual policy process and evaluator:

```bash
MUJOCO_GL=egl uv run advaitic-eval \
  --policy submission.policy:make_policy \
  --smoke --episodes 1 --max-steps 10 --out results/smoke
```

The example only holds the robot's pose. **It does not solve the task** and a smoke result is non-qualifying.

## Implement your method

Put your solution under `submission/`, with a factory returning a callable:

```python
def make_policy():
    return YourPolicy()

class YourPolicy:
    def reset(self):
        pass  # clear episode-specific state

    def __call__(self, obs):
        # obs["rgb"]: uint8 (3, 3, 84, 84), camera then CHW
        # cameras: head, right wrist, left wrist
        # obs["state"]: float32 (50,), robot proprioception
        return actions  # (20,) or (H,20), 1 <= H <= 50
```

The evaluator calls `reset()` at each episode start and `__call__(obs)` when it needs the next action or chunk. You choose the method behind that interface.

Actions are finite normalized values in `[-1,1]`. The first four slots command base forward/lateral velocity, pelvis height and yaw rate; the next fourteen command arm joint targets, and the last two command grippers. [`action_layout.json`](src/advaitic_eval/action_layout.json) contains the exact raw limits. Details are in [METHODS.md](METHODS.md).

No seed, slot index, plate/rack pose, simulator state, reward or success flag is passed to your evaluated policy. Estimate task state from observations. Fixed robot geometry/calibration is allowed. You may use privileged state in a separately disclosed training or annotation workflow, never in final evaluated inference.

The policy runs in a separate process. This keeps candidate imports and normal monkey-patching out of the evaluator's simulator. **It is a process boundary, not a security sandbox.** Reviewers must rerun using a clean trusted evaluator; candidate-generated logs alone are not authoritative.

For a separate model environment, pass its Python path using `--policy-python /absolute/path/to/model-env/bin/python`. That environment needs your method dependencies and NumPy; the runner supplies the worker module on `PYTHONPATH`. No hosting is required.

## Data and training

BiGym2 provides 60 human demonstrations of `move_plate`:

```bash
uv run bigym-download --task move_plate
```

```python
from bigym.loco import make_gym
env = make_gym("move_plate")
demos = env.get_demos(60)
# Use decode_images=False when indexing large datasets without loading images.
env.close()
```

You may use additional data or models with attribution. Provide your training/data-generation commands and configs, not only final weights. Do not train on evaluation seeds or tune to the final reviewer seed block.

## Evaluate

```bash
MUJOCO_GL=egl uv run advaitic-eval \
  --policy submission.policy:make_policy --method my-method \
  --checkpoint my-checkpoint-identifier \
  --episodes 30 --out results/final
```

Use a **new** output directory for each run. The result contains `report.json`, `episodes.csv`, per-episode action arrays and policy stderr. It includes slot coverage, successes, failures, termination causes, simulation time, policy wall time, dependency/controller identity and a submission digest. A run can only pass if it is complete, balanced, integrity-valid and reaches 90%.

To record the first three episodes:

```bash
uv sync --locked --extra video
MUJOCO_GL=egl uv run advaitic-eval \
  --policy submission.policy:make_policy --episodes 30 \
  --video-episodes 3 --out results/final-with-video
```

Environment time advances only when actions execute. Model inference wall time is reported separately; do not present simulator success as proof of real-time throughput.

Source slot means the sampled slot **before** the unchanged 200-step settling warmup; contact settling can shift the plate centre. We record the sampled slot without changing the reset or physics.

## Deliverables

Share your **solution repository URL and exact commit SHA** directly with Advaitic, plus:

1. Exact setup, training and evaluation commands, including model/checkpoint retrieval.
2. A baseline and at least one substantive improvement, with an ablation or comparison.
3. A full qualifying evaluation report and per-slot results; include unsuccessful runs in your experiment log.
4. A short video showing multiple initial slots and at least one failure/recovery when applicable.
5. A concise report explaining architecture, observations/actions, data, experiments, failure modes and next steps.
6. Attribution for borrowed code, pretrained models, datasets and AI-tool use.

Do not submit your solution as a pull request to this starter repository: that exposes it to other candidates. A public fork also exposes your solution. To keep it private, clone the starter into a separate private repository and grant Advaitic access.

## What does not satisfy this assignment

- A fixed open-loop motion script, recorded-trajectory replay, or per-seed/slot solution lookup.
- Copying an existing solution or using an agent to generate one without demonstrating independent understanding, closed-loop behaviour, reproducible improvements and failure analysis.
- Changing evaluation physics, assets, controller weights, resets, time limits, success checks or the evaluator.
- Reading live simulator/task state from outside the permitted observation interface.
- Claiming a success rate from a selected subset or edited result files.

Classical feedback control, state machines and calibrated geometric constants are allowed. **Repeatability and observation-driven decisions matter; a particular coding style or a mandatory neural component does not.**

## Review

We will review the 90% evaluator target above and discuss your implementation and results in a technical deep dive. Only `submission/`, candidate method dependencies and candidate documentation should change. Request clarification for a suspected environment bug; do not silently fix it in your scored run.

## Attribution

Built on [SWIRL BiGym2](https://github.com/swirl-uk/BiGym2), pinned to `d71501359b215b9d78e27dcb70b220412d474dfa`. BiGym2 supplies the task/assets, Unitree G1 model, Dex1-1 grippers and NVIDIA GR00T-WBC controller. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The upstream project publishes agent-written policies; reusing them must be disclosed and is a baseline, not an original submission.
