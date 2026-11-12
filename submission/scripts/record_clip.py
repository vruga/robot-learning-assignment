"""Record one in-process episode (same frame layout as the evaluator's video) for the report.

Illustrative clip only, not an evaluator result. Toggle the stall breaker with BC_STALL_CALLS (0 = off).

    BC_STALL_CALLS=0 python submission/scripts/record_clip.py 7520101 media/seed7520101_no_breaker.mp4
"""

import os
import subprocess
import sys

import numpy as np


def main(seed, out):
    from advaitic_eval.environment import make_environment, observation, observed_source_slot
    from submission.bc_policy import make_policy

    env, pol = make_environment(), make_policy()
    ts = env.reset(seed=seed)
    pol.reset()
    proc = None  # frames are piped to ffmpeg (set IMAGEIO_FFMPEG_EXE or have ffmpeg on PATH)
    n = 0
    while not ts.last():
        for a in np.asarray(pol(observation(ts)), dtype=np.float32).reshape(-1, 20):
            ts = env.step(a)
            n += 1
            if n % 5 == 0:
                frame = np.ascontiguousarray(
                    np.concatenate([c.transpose(1, 2, 0) for c in ts.rgb_obs], axis=1)
                )
                if proc is None:
                    h, w = frame.shape[:2]
                    cmd = [
                        os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg"),
                        "-y",
                        "-loglevel",
                        "error",
                    ]
                    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "10"]
                    cmd += ["-i", "-", "-pix_fmt", "yuv420p", out]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                proc.stdin.write(frame.tobytes())
            if ts.last():
                break
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    ok = bool(env.episode_succeeded() and not env.episode_fell())
    print(
        f"seed {seed} slot {observed_source_slot(env) + 1} success={ok} steps={n} kicks={pol.total_kicks}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
