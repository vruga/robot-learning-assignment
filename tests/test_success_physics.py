"""Real MuJoCo predicates checked against an upstream human demonstration.

These tests inject fixture states to test the evaluator, never to run or
claim a task-solving policy. Scored evaluation only accepts actions.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from advaitic_eval.environment import make_environment, observed_source_slot
from advaitic_eval.protocol import make_plan


@pytest.fixture(scope="module")
def env():
    value = make_environment()
    yield value
    value.close()


@pytest.fixture(scope="module")
def endpoints():
    with np.load(Path(__file__).parent / "fixtures" / "move_plate_endpoints.npz") as file:
        return {key: file[key] for key in file.files}


def apply_state(env, endpoints, frame):
    env.reset(seed=int(endpoints["move_plate__seed"]))
    inner = env.inner_env
    inner.data.qpos[:] = endpoints[f"move_plate__qpos_{frame}"]
    inner.data.qvel[:] = endpoints[f"move_plate__qvel_{frame}"]
    mujoco.mj_forward(inner.model, inner.data)
    inner._step_cache.clean()
    return inner


@pytest.mark.integration
def test_real_reset_and_real_success(env, endpoints):
    assert not apply_state(env, endpoints, "first")._success()
    assert apply_state(env, endpoints, "last")._success()


@pytest.mark.integration
def test_accepts_all_destination_slots(env, endpoints):
    for site_index in range(6):
        inner = apply_state(env, endpoints, "last")
        plate = inner.plates[0]
        sites = [inner.data.bind(site).xpos.copy() for site in inner.rack_target.sites]
        nearest = np.argmin([np.linalg.norm(plate.get_position() - site) for site in sites])
        offset = sites[site_index] - sites[nearest]
        position = plate.get_position() + offset
        quaternion = plate.get_quaternion()
        # Keep the hands away: the recorded final hand pose happens to
        # touch a plate translated into slot 2, which is correctly rejected.
        # This test isolates destination-slot acceptance; holding is tested
        # separately below, without loosening the predicate.
        inner.data.qpos[:] = endpoints["move_plate__qpos_first"]
        inner.data.qvel[:] = 0
        plate.set_position(position, True)
        plate.set_quaternion(quaternion, True)
        inner.simulation.forward()
        inner._step_cache.clean()
        assert inner._success(), f"Destination slot {site_index + 1} rejected"


@pytest.mark.integration
def test_rejects_wrong_orientation_and_off_slot(env, endpoints):
    inner = apply_state(env, endpoints, "last")
    plate = inner.plates[0]
    plate.set_position(plate.get_position() + np.array([0.06, 0, 0]), True)
    inner.simulation.forward()
    assert not inner._success()
    inner = apply_state(env, endpoints, "last")
    inner.plates[0].set_quaternion(np.array([1.0, 0.0, 0.0, 0.0]), True)
    inner.simulation.forward()
    assert not inner._success()


@pytest.mark.integration
def test_held_plate_is_rejected(env, endpoints, monkeypatch):
    inner = apply_state(env, endpoints, "last")
    monkeypatch.setattr(inner.robot, "is_gripper_holding_object", lambda *args: True)
    assert not inner._success()


@pytest.mark.integration
def test_reset_prediction_matches_every_slot(env):
    first_per_slot = {}
    for episode in make_plan():
        first_per_slot.setdefault(episode.source_slot, episode)
    for slot, episode in first_per_slot.items():
        ts = env.reset(seed=episode.seed)
        assert observed_source_slot(env) == slot
        assert ts.rgb_obs.shape == (3, 3, 84, 84)
        assert not env.episode_succeeded()


@pytest.mark.integration
def test_success_hold_requires_consecutive_full_second(env, endpoints, monkeypatch):
    inner = apply_state(env, endpoints, "last")
    assert inner.success_hold_seconds == 1.0
    assert inner.success_hold_steps * inner.control_step_seconds == pytest.approx(1.0)
    # Isolate timer bookkeeping at a real accepted fixture state. No policy
    # rollout or stability claim: physics is frozen while the real step method
    # samples the unchanged physical success predicate.
    monkeypatch.setattr(inner, "_step_mujoco_simulation", lambda action: None)
    monkeypatch.setattr(inner, "_on_step", lambda: None)
    inner.reset_success_hold()
    for _ in range(inner.success_hold_steps - 1):
        inner.step(np.zeros(1), fast=True)
        assert not inner.success
    predicate = inner._success
    monkeypatch.setattr(inner, "_success", lambda: False)
    inner.step(np.zeros(1), fast=True)
    assert not inner.success
    monkeypatch.setattr(inner, "_success", predicate)
    for _ in range(inner.success_hold_steps - 1):
        inner.step(np.zeros(1), fast=True)
        assert not inner.success
    inner.step(np.zeros(1), fast=True)
    assert inner.success


@pytest.mark.integration
def test_floor_contact_fails_and_table_contact_rejects_success(env, endpoints):
    inner = apply_state(env, endpoints, "last")
    plate = inner.plates[0]
    position = plate.get_position().copy()
    position[2] = 0.0
    plate.set_position(position, True)
    inner.simulation.forward()
    inner._step_cache.clean()
    assert plate.is_colliding(inner.floor)
    assert inner._fail()

    inner = apply_state(env, endpoints, "last")
    plate = inner.plates[0]
    position = plate.get_position().copy()
    position[2] = 0.68
    plate.set_position(position, True)
    inner.simulation.forward()
    inner._step_cache.clean()
    assert plate.is_colliding(inner.table)
    assert not inner._success()


@pytest.mark.integration
def test_table_contact_guard_rejects_otherwise_successful_fixture(env, endpoints, monkeypatch):
    inner = apply_state(env, endpoints, "last")
    assert inner._success()
    plate = inner.plates[0]
    collision = plate.is_colliding
    # Physical table contact is checked separately above. Override only that
    # contact result here so distance/orientation cannot mask a missing guard.
    monkeypatch.setattr(
        plate, "is_colliding", lambda prop: True if prop is inner.table else collision(prop)
    )
    assert not inner._success()
