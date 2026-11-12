# 1. Synthetic expert trajectories

Most of the training data for our best policy is **simulator-generated**: 1,329 of the 1,389 training episodes
(96.7% of the frames) come from a scripted, closed-loop expert that reads the simulator's true state. This page
says exactly what that expert is, what it is allowed to see, how it was debugged, and how good the data is.
The human demonstrations (60 episodes, 3.3% of the frames) are the only non-synthetic data we use.

## Why it exists and how it fits the rules

- The assignment allows privileged state "in a separately disclosed training or annotation workflow, never in final
  evaluated inference". The expert is exactly that: **training-time data generation only**. The evaluated policy
  (`bc_policy.py`) never imports it and receives only `rgb` (3 cameras, 84x84) and the 50-d `state`.
- It is **not** an open-loop script, a recorded replay, or a per-seed lookup (all excluded by the README). Every
  command is computed each step from the current simulator state by inverse kinematics; different racks and plate
  positions give different trajectories.
- Why bother: 60 human demos (10 per source slot) are too few. Human-only training reaches 78% on seeds 7,320,000+
  (execute 20); adding 298 expert episodes lifts the same seeds to 98%.

## What it reads (privileged) and what it writes

| Reads (training-time only) | Writes |
|---|---|
| true plate position and orientation | the same 20-d normalised action the policy uses |
| the plate's nearest-rim collider panel (the part the gripper pinches) | base velocity (x, y), pelvis height held at the observed value |
| the 6 destination-slot site positions of the target rack | 7 left-arm joint targets (right arm held still) |
| the left hand site pose and Jacobian (via a scratch copy of the simulator state) | left gripper open/close |
| the observed proprioception (pelvis position and velocity) | |

## How it moves

Only the **left arm** works (as in the human demos); the base walks and strafes. Inverse kinematics is damped least
squares on the 7 left-arm joints toward a 6-D hand pose, taking its error from the **actual** hand pose each step
(so it integrates away tracking sag), with gain 0.2 and at most 0.02 rad per step. The hand keeps its reset
orientation (fingers forward, jaws closing along y); while carrying, a small feedback on the plate's yaw
(gain 0.01) keeps the plate within the 20-degree success limit.

| Phase | What happens |
|---|---|
| `ready` | raise the hand clear of the rack (a pose the arm can reach: at least 18 cm out to the side of the pelvis) |
| `walk` | walk to a standoff 0.51 m in front of the plate and line the left hand up with it |
| `settle` | stop and let the walking gait's body sway die out (25 steps) |
| `pregrasp` | hand to a point just in front of the plate's near rim, jaws straddling it |
| `descend`, `advance` | lower to rim height, then slide the open jaws over the near rim |
| `close` | close the gripper (40 steps) |
| `lift` | raise the plate 12 cm out of its slot |
| `carry` | strafe the base until the plate is over the destination slot in y, also centring the arm's reach in x |
| `steady` | stop and wait for the sway to damp (velocity below a threshold) |
| `align` | arm-only fine alignment of the plate onto the slot, using feedback on the plate pose |
| `place`, `release` | lower into the slot, open the gripper |
| `retreat`, `wait` | back the hand off, hold until success has persisted |

Every approach and transit phase also has a time limit so no phase can deadlock.

## What we learned while debugging it (these are why it reaches 99%)

1. **Grasp the plate's near rim at mid-height**, as the human demos do. Forward kinematics of the human joint states
   showed the hand about 0.43 m ahead of the pelvis near plate-centre height. My first attempt (top rim, arm fully
   stretched) pulled the pelvis backwards and dragged the plate.
2. **Aim at the rim collider, not the plate origin.** The plate is tilted about 5 degrees, so its origin is about 1.5 cm
   off the rim; the jaw gap is only 39 mm around a 10 mm rim, so the first grasps closed on nothing.
3. **"Hold" must freeze the joint command.** Re-anchoring the goal at the current hand made orientation error ratchet
   the arm into a drift. Hold now means no IK update.
4. **IK gain 0.6 caused a limit cycle** (about 0.6 s period, plate yaw swinging 20 degrees). Gain 0.2 fixed it.
5. **Walking sways the body about 5 cm**, so fine alignment cannot be done by the base. The base does the coarse move
   and the arm does the fine alignment with feedback on the plate.
6. **Three failure modes took the first version from 81% to 99.6%:** the raised-hand pose was unreachable near the
   body centreline (deadlock in `ready`); racks shifted far forward put the arm at full reach, so the base now
   centres the reach on the slot; and a plate unusually close to the robot stalled the hand 1.4 cm short of the rim,
   fixed by timeouts with a slightly wider tolerance.

## How good is the data

| Generation run | Seeds | Success |
|---|---|---|
| first full version | 400 | 324/400 (81%); slot 6: 33/57 (58%) |
| final version, fresh seeds | 240 (5,000,400-639) | **239/240 (99.6%)** |
| more data, unfiltered | 640 (5,001,000-639) | 634/640 (99.1%) |
| more data, slots 4-6 only | 400 (from 5,002,000) | 397/400 (99.3%) |

Only successful episodes are kept: 298 + 634 + 397 = **1,329 episodes**, about 1.09M frames (mean length 823 steps
versus 627 for the human demos), stored as `ep_<seed>.npz` with `rgb (T,3,3,84,84)`, `state (T,50)`, `action (T,20)`,
`seed` and `slot`.

**Integrity check** (`scripts/verify_data.py`): replaying the recorded actions of four expert episodes through the
real simulator reproduced every recorded state exactly (max difference 0.0) and each ended in success without a
fall. Replay is a data check only; it is never used as a policy. Expert seeds (5,000,000-5,002,xxx) are disjoint
from every evaluation block (7,310,000 and up).

**It is not the same style as the humans:** the expert holds the right arm perfectly still (humans move it a little)
and grips with a different wrist posture (wrist roll about -0.31 vs -0.10, wrist yaw about 0.34 vs 0.01). The policy
trained on the mixture works, but the expert's habits are what it mostly imitates.

## Reproduce

```bash
MUJOCO_GL=egl uv run python -m submission.generate_data --start-seed 5000000 --count 640 --workers 12 --out datasets/expert_train
# only slots 4-6 (picked without simulating, via the protocol's own slot function): --seeds <list>
MUJOCO_GL=egl .venv/bin/python submission/scripts/verify_data.py --demos datasets/move_plate.npz --expert-dir datasets/expert_train
```

## Limits and a lesson

- The policy can only be as good as the expert's strategy; it inherits the expert's habits and blind spots.
- A DAgger attempt (student rollouts labelled by this expert) made things worse: the expert keeps a joint-command
  integrator, and I overwrote it with the student's executed command, so labels became "what the student just did plus a
  tiny nudge". A correct version keeps the expert's own command and only perturbs what is executed (DART). That
  fixed version is designed but not run.
