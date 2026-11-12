# Experiment log

Target: ≥90% over ≥30 balanced trials (27/30).
**Shipped model: `cr_v5` (+ observation-only stall breaker); see section E/F at the end: 598/600 and 2970/3000 on fresh seeds (99.1% pooled), 30/30 on the public block. The earlier `cr_all` figures below are kept as history.**
The earlier best (298 expert episodes, 60k steps) is kept as the A2 rows.
**Best configuration: human demos + 298 privileged-expert episodes, ResNet18 chunk policy, execute 20 steps per
chunk.** Evaluator-run results from our own runs (not independently re-run by a reviewer):

| Block (seeds) | Trials | Success | Required | Per slot (1..6) | Submission digest |
|---|---|---|---|---|---|
| public 7,310,000+ | 30 | **27 (90.0%)** | 27 | 5,5,5,4,3,5 | db1c3be20cd0 |
| fresh 7,320,000+ | 60 | 59 (98.3%) | 54 | 10,10,10,9,10,10 | 8e59c44eacb6 (earlier snapshot, same policy logic) |
| untouched 7,330,000+ | 60 | **57 (95.0%)** | 54 | 10,10,10,9,9,9 | db1c3be20cd0 |
| fresh 7,340,000+ | 60 | 59 (98.3%) | 54 | 10,10,10,10,10,9 | A2 checkpoint, execute 20, CPU inference |
| fresh 7,350,000+ | 60 | 57 (95.0%) | 54 | 10,10,9,9,9,10 | same |
| pooled | 270 | 259 (95.9%) | – | – | – |

All 7 misses are timeouts; 0 falls, 0 policy errors. Mean success length ≈ 763 of 1,700 steps.
Caveats: execute-20 was chosen after looking at the public block (so its 27/30 is a selected number); it was
then confirmed on two other blocks. 30 trials at exactly 90% has a wide interval (about 74-97%); the 270-trial
pool is the better estimate (Wilson 95%: about 93-98%). A reviewer must rerun on a clean trusted copy.

## A. Evaluated policies (image + proprioception only)

All runs use the public protocol with 5 (or 10) trials per source slot; unless noted seeds 7,310,000+.

| # | Method | Training data | Execute | Trials | Success | Per slot | Notes |
|---|---|---|---|---|---|---|---|
| A0 | Hold-pose interface example | none | – | smoke | 0 | – | not a task attempt |
| A1 | BC baseline, ResNet18 x3 cams + state, 20-step chunks, 30k steps | 60 human demos | 10 | 30 | 15 (50%) | 1,3,3,4,2,2 | 15 timeouts |
| A2 | BC + expert data, 60k steps | 60 human + 298 expert | 10 | 30 | 22 (73%) | 4,5,3,3,4,3 | 8 timeouts |
| A2-e5 | same checkpoint as A2 | same | 5 | 30 | 15 (50%) | 3,3,5,2,1,1 | |
| A2-e20 | same checkpoint as A2 | same | **20** | 30 | **27 (90%)** | 5,5,5,4,3,5 | **best** |
| A2-e30/e50 | same checkpoint | same | 30 / 50 | 30 | 27 / 27 | identical to e20 | **clamped to 20**: the model's chunk is only 20 steps, so these are the same run as e20 |
| A3 | pi0.5 zero-shot (`lerobot/pi05_base`) | none for this robot | 50-step queue | 30 | 0 (0%) | 0,0,0,0,0,0 | un-adapted action space; reference only |
| A4 | DAgger round 1: A2 warm-start + 150 expert + 192 DAgger eps, 40k steps | human + expert + DAgger | 10 | 30 | 17 (57%) | 2,3,3,3,4,2 | worse than A2 |
| A4-e20 | same as A4 | same | 20 | 30 | 23 (77%) | 4,4,5,3,4,3 | worse than A2-e20 (27) |
| A4-devA | same as A4 | same | 20 | 60 (7,320,000+) | 50 (83%) | 10,8,10,8,9,5 | slot 6 weakest |
| A2-devA | A2 checkpoint | same | 20 | 60 (7,320,000+) | 59 (98%) | 10,10,10,9,10,10 | |
| A5 | Same data/architecture as A2 but **50-step chunks** (`bc_h50`), 60k steps | human + 298 expert | 20 | 60 (7,320,000+) | 57 (95%) | 10,9,10,10,9,9 | |
| A5-e35 | same as A5 | same | 35 | 60 (7,320,000+) | 59 (98%) | 10,10,10,10,10,9 | |
| A5-e50 | same as A5 | same | 50 | 60 (7,320,000+) | 59 (98%) | 10,10,10,10,10,9 | |
| A6 | **Human demos only**, continuous head (A1 weights) | 60 human only | 10 | 60 (7,320,000+) | 39 (65%) | 7,5,7,8,7,5 | real data only |
| A6-e20 | same as A6 | same | 20 | 60 (7,320,000+) | 47 (78%) | 5,9,8,9,10,6 | expert data adds ~+20 points (59/60) |
| A2-e5-devA | A2 checkpoint (continuous) | human + 298 expert | 5 | 60 (7,320,000+) | 29 (48%) | | like-for-like baseline for A7 |
| A2-e10-devA | A2 checkpoint (continuous) | same | 10 | 60 (7,320,000+) | 47 (78%) | | |
| A7 | **Discrete head** (paper arXiv:2610.20429 idea, simplified: MLP head, 256 bins, soft-CE tau=0.0025, greedy argmax) | human + 298 expert, 60k steps | 5 | 60 (7,320,000+) | 8 (13%) | | much worse |
| A7-e10 | same as A7 | same | 10 | 60 (7,320,000+) | 13 (22%) | | |
| A7-e20 | same as A7 | same | 20 | 60 (7,320,000+) | 21 (35%) | | 27 timeouts, 12 drops/floor touches |
| A7-cfg1 | A7 + classifier-free guidance, lambda=1 | same | 20 | 60 (7,320,000+) | 1 (2%) | | guidance made it collapse |
| A7-cfg2 | A7 + classifier-free guidance, lambda=2 | same | 20 | 60 (7,320,000+) | 0 (0%) | | |
| A8 | **`cr_all`**: continuous head, 60 human + 1,329 expert eps, slots 4-6 oversampled 2x, 100k steps | human + 1,329 expert | 20 | 30 (public) | 29 (96.7%) | 5,5,5,4,5,5 | **current best** |
| A8-b | same as A8 | same | 20 | 60 (7,340,000+) | 60 (100%) | 10,10,10,10,10,10 | |
| A8-c | same as A8 | same | 20 | 60 (7,350,000+) | 60 (100%) | 10,10,10,10,10,10 | |
| A8-d | same as A8 | same | 20 | 60 (7,360,000+) | 58 (96.7%) | 10,10,9,10,10,9 | |
| A8-e | same as A8 | same | 20 | 60 (7,370,000+) | 60 (100%) | 10,10,10,10,10,10 | pooled A8: 267/270 (98.9%) |
| A9 | **Paper-style discrete flow** (DiT 12 blocks, 512 bins, DF alpha=t^3, 20 flow steps), absolute actions, 60k steps | human + 298 expert | 20 | 60 (7,340,000+) | 55 (91.7%) | 10,10,9,8,10,8 | vs 59/60 continuous |
| A9-b | same as A9 | same | 20 | 60 (7,350,000+) | 53 (88.3%) | 7,10,9,9,9,9 | vs 57/60; pooled 108/120 vs 116/120, p = 0.067 |
| A9-e5 | same as A9 | same | 5 | 60 (7,320,000+) | 48 (80.0%) | 6,9,8,10,7,8 | **continuous: 29/60 at execute 5 (p = 0.0005)** |
| A9-e10 | same as A9 | same | 10 | 60 (7,320,000+) | 50 (83.3%) | 9,9,9,8,8,7 | continuous: 47/60 at execute 10 |
| A10 | pi0.5 fine-tune (action expert only, vision frozen, 5k steps, batch 8, 160 eps) | 60 human + 100 expert | 50 | 30 (public) | 0 (0%) | 0,0,0,0,0,0 | valid run, all timeouts; far too little training |
| A2-final | A2 checkpoint, final defaults | same | 20 | 60 (7,330,000+) | 57 (95%) | 10,10,10,9,9,9 | untouched block |
| A2-public | A2 checkpoint, final defaults | same | 20 | 30 (public) | 27 (90%) | 5,5,5,4,3,5 | |

Findings
- **Execute length is the biggest knob for this checkpoint**: 5 → 50%, 10 → 73%, 20 → 90% (same weights).
  Frequent re-planning from a single-frame policy makes it dither between modes; committing to a full 20-step
  chunk removes that. Longer chunks do not add more: the 50-step-chunk model (A5) scores 57/59/59 of 60 at
  execute 20/35/50 versus 59/60 for the 20-step-chunk model on the same block, i.e. no difference within noise.
- **Expert data is the biggest data lever**: 60 human demos → 15/30 at execute 10; adding 298 expert episodes →
  22/30 (val L1 0.0131 → 0.0040).
- **DAgger round 1 did not help** (A4 below A2 at both execute lengths, and 50/60 vs 59/60 on the fresh block).
  Not yet diagnosed. Candidates: only 150 of the 298 clean expert episodes were kept in the mix (RAM limit);
  student-driven windows labelled by an expert whose IK integrator was re-seeded from the executed command
  may give labels slightly inconsistent with the clean data; no re-tuning of the learning rate for the warm start.
- **Discrete action head did not help** (A7): 35% vs 98% for the continuous head at execute 20, and worse at every
  execute length; guidance (the paper's own fix) collapsed it. This tests a *simplified* version (small MLP head,
  256 bins, one training run, no discrete-flow iteration), so it does not refute the paper. The paper itself
  says plain discrete regression needs guidance or discrete flow to match continuous heads.
- **Real vs synthetic data**: human-only 78% (execute 20) vs 98% with 298 expert episodes added (same seeds).
  Human demos are 13.3% of training frames (37,645 of 282,994). Replaying 4 recorded expert episodes through the
  simulator reproduced the recorded states exactly (max diff 0.0) and each ended in success, no falls; expert
  seeds (5,000,015-5,000,639) are disjoint from every evaluation block. The expert's style differs from the human
  demos (right arm held still, different wrist posture, longer episodes: 823 vs 627 steps).
- **Discrete flow is far more reactive** (the paper's mechanism holds here): re-planning every 5 steps it scores
  48/60 vs 29/60 for the continuous head (p = 0.0005); at execute 10 it is 50/60 vs 47/60 (n.s.); at execute 20,
  where the continuous head is saturated, it is 108/120 vs 116/120 (p = 0.067, not better). Reactivity helps
  where the continuous head dithers, but our best continuous setting does not need it.
- pi0.5 zero-shot is a **reference**, not a real VLA baseline (see D).
- Policy inference: ~0.3 s/episode (ResNet) vs 15.8 s/episode (pi0.5 3B).

## B. Privileged expert (training-time data generator; NOT an evaluated policy)

Reads true plate/rack poses from the simulator. Dev seeds 5,000,000+ (never the evaluation blocks).

| # | Expert version | Seeds | Success | Notes |
|---|---|---|---|---|
| B1 | v0 (first full pipeline) | 12 | 12/12 | too few to be informative |
| B2 | v0 | 400 | 324/400 (81%) | slot 6: 33/57 (58%); 76 timeouts, 0 falls |
| B3 | v1: reachable "ready" pose + phase timeouts | the 76 B2 failures | 8/76 | deadlock in `ready` was only one cause |
| B4 | v2: + base centres the arm's reach on slot x | the 76 B2 failures | 59/76 | racks shifted far forward |
| B5 | v3: + timeouts on approach phases | 240 fresh (5,000,400-639) | **239/240 (99.6%)** | slots 34/34, 45/45, 51/51, 35/35, 46/47, 28/28 |

Failure causes found, in order: (1) raised-hand pose infeasible near the body centreline → deadlock in `ready`;
(2) rack far forward → arm at full reach, plate stalls ~3 cm short; (3) plate close to the robot → hand stalls
1.4 cm short of the rim and an 8 mm tolerance is never met.

Design lessons: grasp the plate's **near rim at mid-height** (as the human demos do); aim at the plate's actual
collider (the plate is tilted ~5°, so its origin is 1.5 cm off the rim); "hold" must freeze the joint command
rather than re-anchor the goal at the current hand; IK gain 0.6 caused a limit cycle (0.2 fixed it); gait sway
(~±5 cm) rules out fine alignment by the base, so the base does the coarse strafe and the arm does fine
alignment with feedback on the plate pose.

## C. Process / validity incidents (kept on purpose)

| What happened | Effect | Resolution |
|---|---|---|
| First A1 run done while I synced files into `submission/` | `submission_unchanged=false`, invalid (15/30, indicative only) | rerun |
| Second A1 run overlapped the A2 training job | still `submission_unchanged=false`: the evaluator hashes `checkpoints/*.pt`, which training rewrites every 1k steps | evals re-run only after training ended; later evals use isolated snapshot dirs via `--policy-root` |
| Expert code was debugged on seed 7,310,000 (public block) | tuning-leakage risk for the expert only | all later development and data use seeds 5,000,000+ / 6,000,000+; student never trained on 7,3xx,xxx |
| execute length chosen after viewing the public block | selected number on that block | confirmed on two untouched blocks |
| Evals launched while the GPU was full: policy process hit CUDA OOM, 60 `policy_error` episodes at 0 steps | invalid "0/60" records (kept as `INVALID_oom_*`) | policy inference moved to CPU via `BC_DEVICE=cpu` for the small ResNet policies |
| Disk filled twice (shared 4090): a DR training save was truncated, a pi0.5 job lost its log | corrupt checkpoint; relaunch | datasets/envs moved to the big volume; checkpoint saves made atomic |
| Disk filled (shared 4090) during installs | one env install failed | deleted only my own redundant data |
| `generate_data.py` and `dagger.py` refuse to run on 7,310,000-7,319,999 | – | – |

## D. Not done / next

- G1-specific VLAs: a community pi0.5 G1 checkpoint (EmbodyX) is not accessible (404/401); Unitree's UnifoLM-VLA-0
  (19 GB, Qwen2.5-VL + diffusion head, 7-d action/state, wrist cameras, CC BY-NC-SA) needs its own codebase, a
  separate VLM backbone, LeRobot→HDF5→RLDS conversion and a 4.7B-parameter fine-tune; not attempted.
- pi0.5 fine-tune (action expert only, frozen vision tower, bf16, 160 episodes = 60 human + 100 expert) is
  in progress; first evaluation at step 5,000.
- **VLA comparison is not a real baseline yet**: pi0.5 would need fine-tuning on the expert data (LeRobot
  dataset conversion + training); zero-shot 0/30 only shows the base model does not transfer.
- Diagnose why DAgger round 1 hurt; try lower learning rate, all clean expert data, relabel-only (beta 0).
- Ablations not done: data size (60 / 150 / 300 expert episodes), pretrained ResNet, seeds of training.
- Independent clean-copy rerun by a reviewer is required for a qualifying claim.

## Reproduce

See `PROBLEM.md`. Final configuration:
`BC_CHECKPOINT=checkpoints/bc_expert/last.pt BC_EXECUTE=20 uv run advaitic-eval --policy submission.bc_policy:make_policy --policy-python <torch env> --episodes 60 --seed-start 7330000 ...`

## E. Final improvement round (Nov 11): failure diagnosis, stall breaker, DART + hard-config data

Evaluator-run results from our own runs; fresh seeds were never used for training or tuning.

| Run | Seeds | Trials | Success | Notes |
|---|---|---|---|---|
| cr_all (no stall breaker) | 7,340,000+ and 7,350,000+ | 120 | 120 (100%) | earlier selection blocks |
| cr_all (no stall breaker) | 7,360,000+ / 7,370,000+ | 300 | 296 (98.7%) | 4 timeouts: 7360022, 7360025, 7360087, 7360139 |
| cr_all + stall breaker | 7,360,000+ | 150 | 147 (98.0%) | breaker fixed 7360087; **dev block, misses were used for diagnosis** |
| cr_v5 + stall breaker | 7,360,000+ | 150 | 149 (99.3%) | dev block; fixed 3 old misses, new miss 7360121 |
| **cr_all + stall breaker** | 7,500,000-7,530,000 (4x150) | 600 | 594 (99.0%) | Wilson 95%: 97.8-99.5%; slots 100,99,99,100,99,97 |
| cr_v5 + stall breaker v1 (shipped at the time) | same 600 seeds | 600 | 597 (99.5%) | Wilson 95%: 98.5-99.8%; **optimistic: superseded by the 3,000-trial run below (99.0%)** |

Pre-registered rule (`reports/prereg/PREREG_final600.txt`): ship cr_v5 if it has no more misses than cr_all on the same fresh seeds
(3 vs 6). The difference is **not statistically significant** (Fisher p about 0.5).

Diagnosis of the four 7,36x misses (replay of the saved actions with read-only privileged probes,
`scripts/diagnose_miss.py`): two frozen fixed points (gripper closed but not lifting; hovering ~6 cm from the slot),
one release ~5 cm off the slot, one drop. The deterministic single-frame policy had no learned way out of states
the clean expert data never visits.

Changes: (1) observation-only stall breaker in `bc_policy.py` (arm/gripper joints unchanged for 5 replans -> noise on the
proprioceptive input, sigma 1.0 growing per repeat); 0 false triggers on 120 normal dev episodes; fixes the
post-grasp freeze, not the hover/early-release/drop cases. (2) New expert data: 314 clean episodes on hard rack
configurations (large x offset, slots 3-6; seeds 5,003,000+), and 396 DART episodes (brief noise bursts on base and
left-arm commands, expert keeps its own command integrator so labels stay clean; seeds 5,004,000+; constant noise
broke the expert's phase switches: 8/12 at sigma 0.005-0.015, 2/12 at 0.01-0.03, bursts 12/24 at 0.03-0.08).
(3) Retrained the cr_all recipe, 150k steps, on 60 human + 710 new + 200 old expert episodes (held-out L1 0.00289;
split differs from cr_all's 0.0028). Default weights are `submission/weights/final.pt`.
Not isolated: whether the gain comes from DART, from hard configs, or from more steps.

### F. Larger certification (Nov 12): the 99.9% goal was NOT reached

Pre-registered in `reports/prereg/PREREG_round2.txt` before any result. Stall breaker v2 (3 stalled replans, random-strength kicks;
on 120 healthy dev episodes it fired in 1 (a slot-6 freeze it rescued; 120/120 succeeded) (run with the in-process `scripts/probe_seeds.py`, not the evaluator; all scores in the tables are `advaitic-eval` runs via `logs/snap_eval.sh`); rescues the post-grasp freeze of 7520101 and 7360087 at normal speed).

| Run | Seeds | Trials | Success | Notes |
|---|---|---|---|---|
| cr_v6 (+294 clean slot-3/6 hard, +376 DART bursts in carry/place) | 7,600,000-7,630,000 (4x150) | 600 | 592 (98.7%) | 8 misses, 3 of them 'terminated'; **worse** than cr_v5 on the same seeds, not shipped |
| cr_v5 + stall breaker v2 | same 600 seeds | 600 | 598 (99.7%) | screening |
| **cr_v5 + stall breaker v2 (shipped)** | 7,700,000 + k*10,000, k=0..9 (10x300) | 3,000 | **2,970 (99.0%)** | Wilson 95%: 98.6-99.3%; slots 499,500,500,495,495,481; 29 timeouts + 1 terminated, 0 falls/errors |

A first launch of this stage used overlapping seed ranges (162 duplicate seeds); it was stopped and deleted before any
result and relaunched with disjoint blocks.

Reading: the earlier 99.5% (600 trials) was a favourable sample; the 3,000-trial estimate is 99.0%. 19 of the 30
misses are slot 6 (481/500 = 96.2%), 9 are slots 4-5. Replaying four slot-6 misses: the plate is carried to within
2-3 cm of the slot, then dropped and left lying flat at the wrong height. That is a placement-time drop, not a freeze,
so the stall breaker cannot help and a dropped plate is not recoverable. Adding more perturbation data in the place phase
(cr_v6) made things worse, so the remaining error is not fixed by simply adding more of this data.

Follow-up probe (in-process `probe_seeds.py`, six slot-6 hover/drop seeds, not an evaluator result): the shipped stall
breaker's strong kicks can open the gripper mid-hold and drop the plate (all six fail with it on); breaker off passes 2/6;
a gentler fixed kick (sigma 0.3, 3 stalled replans) passes 3/6. Any change based on this needs an evaluator run on fresh seeds.

### G. Gripper-hold stall breaker (Nov 12)

Replay of 6 slot-6 / near-slot-6 misses (`diagnose_miss.py`): 5 of 6 held the plate 3-6 cm from the slot, froze for 80-220
steps, then the gripper opened (0.9 -> 0.5 -> 0) with the plate above the slot and it dropped. Change: `BC_HOLD_GRIP`
(now default on): while a stall-breaker kick is active and the gripper is closed, keep the previous gripper command.
Pre-registered in `reports/prereg/PREREG_hold.txt`. Dev (32 known-miss seeds, in-process, biased selection): shipped 17/32, hold 28/32.
Fresh screen, seeds 7,800,000 + k*10,000 (k=0..3), 4x150, identical seeds, same weights:

| Config | Success | Misses |
|---|---|---|
| breaker v2 (shipped) | 594/600 (99.0%) | 7810024, 7820030, 7820079, 7820080, 7830116, 7830121 |
| + hold gripper | 598/600 (99.7%) | 7810024, 7830121 |

Rule met (misses hold <= base, no new falls/errors). Not statistically significant (Fisher p about 0.28); a 600-trial screen
only rules out regressions. Certification run: seeds 7,900,000+ (10x300), see `reports/prereg/PREREG_hold.txt`.

Certification result (pre-registered in `reports/prereg/PREREG_hold.txt`, seeds 7,900,000 + k*10,000, 10x300, hold default, weights unchanged):
**2,991/3,000 (99.7%)**, Wilson 95% 99.4-99.8%, per slot 499/500/500/498/498/496, 8 timeouts + 1 early termination (7930139, no fall),
0 falls, 0 policy errors, integrity ok. Misses: 7900129, 7930139, 7940032, 7950209, 7950240, 7950286, 7970183, 7970247, 7980162.
Versus the previous version (2,970/3,000 on 7,7xx seeds): Fisher p about 0.001 (different seeds). 99.9% goal not reached.
Public block with video (7,310,000+): 30/30. Reports: `reports/final_hold/`.
