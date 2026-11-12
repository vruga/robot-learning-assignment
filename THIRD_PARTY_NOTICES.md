# Attribution and upstream components

This assignment depends on SWIRL BiGym2 at commit
`d71501359b215b9d78e27dcb70b220412d474dfa`:
<https://github.com/swirl-uk/BiGym2>.

BiGym2 authors: Zexi Zhang, Zecheng Zhu, Zidong Chen, Zulkhuu Tuya and Stephen James,
Department of Computing, Imperial College London.

The simulator, task/assets, controller and robot models are fetched as the pinned
dependency, not reimplemented or relabelled as Advaitic's original work. Its
licenses and attribution files remain with that dependency. Copies of the upstream
notices are included as `UPSTREAM_NOTICE` and `UPSTREAM_THIRD_PARTY_NOTICES.md`.
In particular, NVIDIA controller weights have their own model license, and robot
meshes and third-party props have their own notices listed upstream.

`tests/fixtures/move_plate_endpoints.npz` is the move_plate-only subset of
BiGym2's `tests/fixtures/demo_endpoints.npz` at the pinned commit. It contains
first/last simulator states of an upstream human demonstration, used only to test
success predicates. It is not a solution policy or evaluation submission.

Reference: Z. Zhang et al., *BiGym 2.0: Benchmarking Learned and Agent-Developed
Policies for Humanoid Household Manipulation*, arXiv:2610.07594, 2026.
<https://bigym2.github.io/>.
