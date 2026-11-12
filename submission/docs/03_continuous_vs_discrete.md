# 3. Continuous vs discrete action heads

We tested the central claim of *"When Do Discrete Actions Make Good Continuous Controllers?"* (Yi et al., arXiv
2610.20429): that discrete action prediction (uniform bins + cross-entropy) can match or beat continuous heads for robot
control, because it is more **reactive** and does not over-smooth. **Short answer for our task: no, discrete is not
better where it matters, but the paper's reactivity mechanism is real.**

## The heads we compared (same image backbone and data unless stated)

| Name | What it is |
|---|---|
| **Continuous (L1)** | our baseline: MLP head regressing the 20x20 chunk with an L1 loss (page 2) |
| **Discrete, simplified** | my first attempt: MLP head, 256 bins per value, cross-entropy with lightly smoothed labels, greedy argmax; optional guidance |
| **Discrete flow (DiT)** | the paper's strongest head, as faithfully as I could build it: 12-block transformer (width 256, 4 heads) over 400 action tokens (20 steps x 20 dims) conditioned through adaptive LayerNorm on the observation; 512 bins; mixture-path corruption with alpha(t)=t^3 from a uniform source; trained on clean-token cross-entropy; 20 refinement steps at inference; classifier-free guidance supported |

Training for the discrete-flow head: 60k steps, batch 128, lr 2e-4, on 60 human + 298 expert episodes (the same data as the
earlier continuous model, whose numbers are the ones compared below).

## Results (60 trials per cell unless noted; our own evaluator runs)

Seeds 7,320,000+, trained on human + 298 expert episodes:

| Re-plan every | Continuous (L1) | Discrete, simplified | Discrete flow (DiT) |
|---|---|---|---|
| 5 steps | 29/60 (48%) | 8/60 (13%) | **48/60 (80%)** |
| 10 steps | 47/60 (78%) | 13/60 (22%) | **50/60 (83%)** |
| 20 steps | 59/60 (98%) | 21/60 (35%) | not run on this block |

Seeds 7,340,000+ and 7,350,000+, execute 20:

| | Continuous (L1) | Discrete flow (DiT) |
|---|---|---|
| 7,340,000+ | 59/60 | 55/60 |
| 7,350,000+ | 57/60 | 53/60 |
| **pooled** | **116/120 (96.7%)** | **108/120 (90.0%)** |

Guidance on the simplified discrete head (execute 20): weight 1 -> 1/60, weight 2 -> 0/60 (it collapsed the policy).
For reference, `cr_all` (continuous, 4x more data) scored 267/270 (98.9%); it is not part of this like-for-like comparison.

Significance (Fisher exact, two-sided): execute 5, 48/60 vs 29/60, **p = 0.0005**; execute 10, 50/60 vs 47/60, p = 0.64;
execute 20 on the two blocks, 108/120 vs 116/120, **p = 0.067** (a gap we cannot call real, but certainly not an
advantage).

Held-out action error (L1, same held-out episodes): continuous 0.0040 (0.0028 for `cr_all`), discrete flow 0.0063,
simplified discrete 0.0084.

## What this says about the paper

- **Confirmed: discrete flow is much more reactive.** Re-planning every 5 steps, the continuous head collapses to 48% (it
  dithers when it re-plans from a single frame) while discrete flow holds 80%. That is the paper's mechanism showing up
  in our task.
- **Not confirmed: discrete beats continuous.** At the setting where the continuous head is saturated (execute 20), discrete
  flow is 90% vs 97%, not better. The benefit exists but our best configuration does not need it.
- **A naive discrete head does not work for us** (35% vs 98%); the paper also says plain discrete regression needs guidance
  or iterative refinement, and it is the iterative version that approaches continuous here.

## Why our result differs from the paper's (what is measured, what is a hypothesis)

| Difference | Status |
|---|---|
| The paper's continuous baseline is squared-error regression, which averages conflicting modes and over-smooths; ours is **L1**, much less prone to that | stated in the paper and in our code; I did not train the squared-error head or run the jerk-ratio diagnostic (`diag_heads.py` is written, not run), so "our L1 head does not over-smooth" is **not proven** |
| Training budget: they use 300k steps at batch 512 (about 154M samples); ours was 60k at batch 128 (7.7M, about 5%) | measured; our discrete-flow error plateaued at 0.0063 |
| Our data is mostly **one deterministic scripted expert**; the paper's gains come from multi-modal human data where averaging modes hurts | hypothesis (consistent with a plain regression head working so well here) |
| The paper predicts **relative** actions (target minus current pose); ours are absolute joint targets, so our 512 bins are about 3-5x coarser for the left arm than relative bins would be | not tested: a relative-action mode is implemented and smoke-tested, never trained |
| Inference: we used 20 flow steps (the paper uses up to 100) and no guidance on the flow head | not tested |
| Sample size: 120 trials cannot separate a 6-7 point gap from noise | statistics above |

## What would settle it (not run)

1. Train the discrete-flow head with **relative** actions, 150k+ steps, on the 1.1M-frame data, and evaluate at execute 5, 10 and 20.
2. Train the paper's **squared-error** continuous baseline and run the jerk-ratio and reactivity diagnostics on all heads.
3. Discrete flow with 50-100 flow steps and with guidance.
4. If a discrete head ever beats continuous at execute 20, it would need hundreds of trials per candidate to show it: the continuous
   head already misses about 1%.

## Reproduce

```bash
.venv-train/bin/python -m submission.train --packed /var/tmp/sra_data/packed --head dit-df --bins 512 --lr 2e-4 --out ckpt/dit_df --steps 60000
BC_FLOW_STEPS=20 BC_CHECKPOINT=ckpt/dit_df/last.pt BC_EXECUTE=5 advaitic-eval --policy submission.bc_policy:make_policy ...   # also 10, 20
# (the head choices are --head continuous | discrete | dit-dr | dit-df; --relative and --ema are the paper-aligned options)
```
