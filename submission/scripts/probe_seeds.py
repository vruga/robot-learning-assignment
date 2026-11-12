"""Run chosen seeds in-process (evaluator-equivalent loop) to test policy variants. Dev use only.

BC_STALL_CALLS=6 .venv-train/bin/python submission/scripts/probe_seeds.py 7360025,7360087
"""

import sys

import numpy as np


def main(seeds):
    from advaitic_eval.environment import make_environment, observation, observed_source_slot
    from submission.bc_policy import make_policy

    env = make_environment()
    pol = make_policy()
    for seed in seeds:
        ts = env.reset(seed=seed)
        pol.reset()
        n = 0
        while not ts.last():
            for a in np.asarray(pol(observation(ts)), dtype=np.float32).reshape(-1, 20):
                ts = env.step(a)
                n += 1
                if ts.last():
                    break
        ok = bool(env.episode_succeeded() and not env.episode_fell())
        print(
            f"seed {seed} slot {observed_source_slot(env) + 1} success={ok} steps={n} "
            f"end={env.episode_termination()} kicks={pol.total_kicks}",
            flush=True,
        )


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")])
