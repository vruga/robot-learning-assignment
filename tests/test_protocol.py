from collections import Counter

import numpy as np
import pytest

from advaitic_eval.environment import validate_actions
from advaitic_eval.protocol import make_plan, plan_digest, summarize
from advaitic_eval.wire import pack_array, unpack_array


@pytest.mark.parametrize("count", [30, 36, 60, 120])
def test_balanced_deterministic_unique_plan(count):
    plan = make_plan(count)
    assert len(plan) == count
    assert len({ep.seed for ep in plan}) == count
    assert Counter(ep.source_slot for ep in plan) == {slot: count // 6 for slot in range(6)}
    assert plan == make_plan(count)
    assert plan_digest(plan) != plan_digest(make_plan(count, 8_000_000))


@pytest.mark.parametrize("count", [0, 6, 24, 29, 31, 100])
def test_incomplete_or_unbalanced_count_refused(count):
    with pytest.raises(ValueError):
        make_plan(count)


def records(successes=27):
    return [
        {"source_slot": ep.source_slot, "completed": True, "success": i < successes}
        for i, ep in enumerate(make_plan())
    ]


def test_threshold_and_nonqualifying_runs():
    assert summarize(records(), 30, scored=True, integrity_ok=True)["status"] == "PASS"
    assert summarize(records(26), 30, scored=True, integrity_ok=True)["status"] == "FAIL"
    for data, scored, integrity in [
        (records()[:29], True, True),
        (records(), False, True),
        (records(), True, False),
    ]:
        assert (
            summarize(data, 30, scored=scored, integrity_ok=integrity)["status"] == "NON_QUALIFYING"
        )
    data = records(30)
    data[0]["source_slot"] = (data[0]["source_slot"] + 1) % 6
    assert not summarize(data, 30, scored=True, integrity_ok=True)["meets_success_criterion"]


def test_failure_trials_stay_in_denominator():
    data = records(27)
    data[-1]["policy_error"] = "invalid action"
    assert summarize(data, 30, scored=True, integrity_ok=True)["success_rate"] == 0.9


def test_36_trials_rounds_required_successes_up():
    data = [
        {"source_slot": ep.source_slot, "completed": True, "success": i < 32}
        for i, ep in enumerate(make_plan(36))
    ]
    result = summarize(data, 36, scored=True, integrity_ok=True)
    assert result["required_successes"] == 33
    assert result["status"] == "FAIL"
    data[32]["success"] = True
    assert summarize(data, 36, scored=True, integrity_ok=True)["status"] == "PASS"


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(19),
        np.zeros((51, 20)),
        [float("nan")] * 20,
        [float("inf")] * 20,
        [1.1] * 20,
        {"actions": [0] * 20},
        "not an array",
        [10**1000] * 20,
    ],
)
def test_invalid_action_fails(bad):
    with pytest.raises(ValueError):
        validate_actions(bad)


@pytest.mark.parametrize("shape,dtype", [((3, 3, 84, 84), "uint8"), ((50,), "float32")])
def test_wire_preserves_observations(shape, dtype):
    value = np.ones(shape, dtype=dtype)
    np.testing.assert_array_equal(value, unpack_array(pack_array(value)))


def test_actions_accept_single_and_chunk():
    assert validate_actions(np.zeros(20)).shape == (1, 20)
    assert validate_actions(np.zeros((10, 20))).shape == (10, 20)
