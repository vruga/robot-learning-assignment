from types import SimpleNamespace

import pytest

from advaitic_eval import runner
from advaitic_eval.integrity import submission_digest
from advaitic_eval.protocol import PUBLIC_SEED_START, source_slot_for_seed


@pytest.mark.parametrize("failure", ["malformed_action", "timeout"])
def test_policy_failures_complete_full_scored_denominator(tmp_path, monkeypatch, failure):
    class Environment:
        def reset(self, *, seed):
            self.slot = source_slot_for_seed(seed)
            return SimpleNamespace(last=lambda: False, rgb_obs=[], low_dim_obs=[])

        def substrate_fingerprint(self):
            return {}

        def episode_succeeded(self):
            return False

        def episode_fell(self):
            return False

        def close(self):
            pass

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def act(self, obs):
            if failure == "timeout":
                raise TimeoutError("timed out")
            return {"not": "an action array"}

        def close(self):
            pass

    monkeypatch.setattr(runner, "make_environment", Environment)
    monkeypatch.setattr(runner, "observed_source_slot", lambda env: env.slot)
    monkeypatch.setattr(runner, "PolicyProcess", Process)
    monkeypatch.setattr(runner, "check_integrity", lambda: {"ok": True})
    # Keep generated action/report files from affecting candidate identity in this unit test.
    monkeypatch.setattr(runner, "submission_digest", lambda root: "fixed-test-identity")
    result = runner.run(
        SimpleNamespace(
            smoke=False,
            max_steps=None,
            episodes=30,
            seed_start=PUBLIC_SEED_START,
            out=tmp_path / "results",
            policy_root=tmp_path,
            policy="candidate:make_policy",
            policy_timeout=0.2,
            policy_python="unused",
            video_episodes=0,
            method="test",
            checkpoint="none",
        )
    )
    assert result["status"] == "FAIL"
    assert result["completed_episodes"] == 30
    assert result["success_rate"] == 0
    assert result["infrastructure_error"] is None
    assert all(ep["termination"] == "policy_error" for ep in result["episodes"])
    assert all(slot["episodes"] == 5 for slot in result["per_source_slot"].values())


@pytest.mark.parametrize(
    "filename", ["uv.lock", "poetry.lock", "requirements.txt", "requirements-model.txt"]
)
def test_submission_digest_covers_dependency_locks(tmp_path, filename):
    dependency = tmp_path / filename
    dependency.write_text("dependency version 1\n")
    before = submission_digest(tmp_path)
    dependency.write_text("dependency version 2\n")
    assert submission_digest(tmp_path) != before
