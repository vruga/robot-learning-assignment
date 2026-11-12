# 2. The `cr_all` imitation-learning baseline (our best model)

`cr_all` is plain behaviour cloning: a small network that looks at the three camera images and the robot's own joint
readings and predicts the next 20 actions. It is the policy we submit. "Baseline" here means the simple, standard
imitation-learning recipe; every improvement over it came from data, not from a cleverer model.

## What the policy sees and does

- **Input** (the only thing the evaluator passes): `rgb` = 3 cameras (head, right wrist, left wrist) at 84x84, and
  `state` = 50-d proprioception. No seed, slot, plate or rack pose, simulator state or reward.
- **Output:** a chunk of 20 actions (20-d each: 4 base commands, 14 arm joint targets, 2 grippers), all in [-1, 1].
  The evaluator executes the first 20 (`BC_EXECUTE=20`, i.e. the whole chunk, 0.4 simulated seconds) and then asks
  again with fresh observations.

## Model (about 14.3M parameters, `model.py: ChunkPolicy`)

ResNet18 (trained from scratch) is shared across the three cameras (ImageNet mean/std normalisation, a learned
per-camera embedding added to the 512-d features) -> concatenated with a 128-d embedding of the state -> MLP
(1664 -> 1024 -> 1024) -> 20x20 outputs through a `tanh`.

## Training (`train.py`)

| | |
|---|---|
| loss | L1 (mean absolute error) on all 20 steps x 20 dims |
| optimiser | AdamW, lr 3e-4 with a one-cycle schedule (5% warm-up), weight decay 1e-4, gradient clip 1.0, bf16 autocast |
| batch / steps | 128 / 100,000 (12.8M samples, about 11 passes over 1.13M frames) |
| weights used | exponential moving average (decay up to 0.999) |
| augmentation | random image shifts of up to 6 pixels per camera; Gaussian noise (0.05) on the normalised state |
| data | 60 human demos + 1,329 expert episodes (see `01_expert_trajectories.md`): 1,127,752 frames, **3.3% human** |
| slot weighting | frames from source slots 4, 5 and 6 are sampled twice as often (the previous model's misses were all there) |
| episode ends | chunks that run past an episode's end repeat its last action |
| held-out error | L1 0.0028 on 6 held-out episodes (previous best 0.0040) |

```bash
.venv-train/bin/python -m submission.train --packed /var/tmp/sra_data/packed_all --head continuous \
  --boost-slots 3,4,5 --boost 2.0 --steps 100000 --out /var/tmp/sra_ckpt/cr_all
```

## Results (our own evaluator runs; every run valid and qualifying; 0 falls, 0 policy errors, misses are timeouts)

| Seeds | Trials | Success | Per slot 1..6 |
|---|---|---|---|
| 7,310,000+ (public) | 30 | **29 (96.7%), PASS (27 needed)** | 5,5,5,4,5,5 |
| 7,340,000+ | 60 | 60 (100%) | 10,10,10,10,10,10 |
| 7,350,000+ | 60 | 60 (100%) | 10,10,10,10,10,10 |
| 7,360,000+ | 60 | 58 (96.7%) | 10,10,9,10,10,9 |
| 7,370,000+ | 60 | 60 (100%) | 10,10,10,10,10,10 |
| **pooled** | **270** | **267 (98.9%)** | |

The three misses are seed 7,310,003 (slot 4), 7,360,022 (slot 6) and 7,360,025 (slot 3), all with the racks shifted
toward their extremes. They are stalls, not slowness: a successful episode takes about 15 of the 34 allowed seconds. For the one miss we examined
(seed 7,310,003, see `../video/README.md`), the policy closes the gripper at about 6 s and then outputs an almost
unchanged command for the remaining 28 s: a fixed point of the policy. The other two misses were not analysed.
Blocks 7,340,000 and later were not used for any decision about this model; the slot weighting came from the earlier
model's failures on blocks up to 7,330,000+. This is our own result; a reviewer must rerun it on a clean copy.

## The ablations that explain the number

| Question | Evidence (same recipe unless noted) |
|---|---|
| Is the synthetic data needed? | human demos only: 65% (execute 10) / 78% (execute 20) on seeds 7,320,000+; adding 298 expert episodes: 98% |
| Does more data help? | 298 expert episodes, 60k steps: 259/270 (95.9%) pooled; 1,329 episodes, 100k steps: 267/270 (98.9%); on the two blocks both were run on: 116/120 vs 120/120 |
| How often to re-plan? | the same weights score 50% / 73% / 90% (public) when executing 5 / 10 / 20 actions per chunk; 48% / 78% / 98% on seeds 7,320,000+. Re-planning from a single frame makes the policy dither |
| Longer chunks? | 50-step chunks: 57/59/59 of 60 at execute 20/35/50, equal to 59/60 for 20-step chunks |
| DAgger? | worse (50/60 vs 59/60), a labelling flaw in my implementation (see page 1) |

## Reproduce and evaluate

```bash
# weights: vruga/g1-plate-bc-expert-xl on Hugging Face, or train as above
BC_CHECKPOINT=checkpoints/cr_all.pt BC_EXECUTE=20 MUJOCO_GL=egl uv run advaitic-eval \
  --policy submission.bc_policy:make_policy --policy-python $PWD/.venv-train/bin/python \
  --method cr_all --checkpoint cr_all_100k --episodes 30 --out results/final
```

Run evaluations from an isolated `--policy-root` (a folder holding only `submission/` and the checkpoint): the
evaluator hashes `checkpoints/*.pt` under its root, so a training job writing there would invalidate the run.
The checkpoint's sha256 is `35e556e8a449782b62f66bb12649a9e3c7048c4f9c5f89b1b5672375176bc7af`.
