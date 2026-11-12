from pathlib import Path

import numpy as np

from advaitic_eval.policy_process import PolicyProcess


def test_worker_observation_boundary_and_reset(tmp_path):
    (tmp_path / "candidate.py").write_text("""
import numpy as np
class Policy:
    def reset(self):
        self.i = 0
    def __call__(self, obs):
        assert set(obs) == {"rgb", "state"}
        assert obs["state"].shape == (50,)
        assert obs["rgb"].dtype == np.uint8
        print("candidate logging must not corrupt protocol stdout")
        self.i += 1
        return np.ones(20) * self.i / 10
def make_policy():
    return Policy()
""")
    worker = PolicyProcess("candidate:make_policy", tmp_path, tmp_path / "stderr.log")
    obs = {"rgb": np.zeros((3, 3, 84, 84), np.uint8), "state": np.zeros(50, np.float32)}
    try:
        worker.reset()
        np.testing.assert_allclose(worker.act(obs), 0.1)
        np.testing.assert_allclose(worker.act(obs), 0.2)
        worker.reset()
        np.testing.assert_allclose(worker.act(obs), 0.1)
    finally:
        worker.close()
    assert "candidate logging" in Path(tmp_path / "stderr.log").read_text()


def test_timeout_bounds_partial_response_and_disables_transport(tmp_path):
    import sys
    import time

    import pytest

    script = tmp_path / "partial-python"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, time\n"
        "os.read(0, 4096)\n"
        "os.write(1, b'{')\n"
        "time.sleep(30)\n"
    )
    script.chmod(0o755)
    worker = PolicyProcess(
        "unused:factory", tmp_path, tmp_path / "stderr.log", python=str(script), timeout=0.2
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.reset()
        assert time.monotonic() - started < 3
        with pytest.raises(RuntimeError, match="unavailable"):
            worker.reset()
    finally:
        worker.close()


def test_timeout_also_bounds_blocked_request_write(tmp_path):
    import sys
    import time

    import pytest

    script = tmp_path / "blocked-python"
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    script.chmod(0o755)
    worker = PolicyProcess(
        "unused:factory", tmp_path, tmp_path / "stderr.log", python=str(script), timeout=0.2
    )
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            worker.request({"op": "act", "payload": "x" * 2_000_000})
        assert time.monotonic() - started < 3
    finally:
        worker.close()


def test_policy_error_response_allows_next_episode_reset(tmp_path):
    import pytest

    (tmp_path / "candidate.py").write_text("""
class Policy:
    def __init__(self):
        self.episodes = 0
    def reset(self):
        self.episodes += 1
    def __call__(self, obs):
        if self.episodes == 1:
            raise ValueError("first episode fails")
        return [0.0] * 20
def make_policy():
    return Policy()
""")
    worker = PolicyProcess("candidate:make_policy", tmp_path, tmp_path / "stderr.log")
    try:
        worker.reset()
        with pytest.raises(RuntimeError, match="Policy raised an error"):
            worker.act({"state": np.zeros(50, np.float32)})
        worker.reset()
        assert worker.act({"state": np.zeros(50, np.float32)}) == [0.0] * 20
    finally:
        worker.close()
