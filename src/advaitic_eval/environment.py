"""Unmodified BiGym2 task; observations only cross the policy boundary."""

from __future__ import annotations

import numpy as np

from .protocol import SLOT_COUNT


def make_environment():
    import onnxruntime

    # The assignment runs locally; no optional runtime telemetry is needed.
    onnxruntime.disable_telemetry_events()
    from bigym.loco import make

    env = make("move_plate")
    inner = env.inner_env
    original_reset = inner._on_reset

    def record_reset_slot():
        original_reset()
        # Metadata only, before the unchanged 200-step settling warmup.
        # Contact settling can move the plate centre toward an adjacent
        # divider; nearest slot AFTER settling is not the sampled slot.
        sites = inner.rack_start.sites
        if len(sites) != SLOT_COUNT:
            raise RuntimeError(f"Expected {SLOT_COUNT} source slots, got {len(sites)}")
        distances = [
            np.linalg.norm(inner.plates[0].get_position() - inner.data.bind(s).xpos) for s in sites
        ]
        if min(distances) > 0.05:
            raise RuntimeError("Plate was not initialized in a source slot")
        inner._advaitic_initial_slot = int(np.argmin(distances))

    inner._on_reset = record_reset_slot
    return env


def observed_source_slot(env) -> int:
    return env.inner_env._advaitic_initial_slot


def observation(timestep) -> dict:
    # No task pose, rack slot, seed, reward, success flag, env or mjData.
    return {
        "rgb": np.asarray(timestep.rgb_obs, dtype=np.uint8),
        "state": np.asarray(timestep.low_dim_obs, dtype=np.float32),
    }


def validate_actions(value, action_dim: int = 20) -> np.ndarray:
    try:
        actions = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Actions must be a numeric array") from exc
    if actions.shape == (action_dim,):
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != action_dim or not 1 <= len(actions) <= 50:
        raise ValueError(f"Expected (20,) or (H,20), 1<=H<=50; got {actions.shape}")
    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.0):
        raise ValueError("Actions must be finite, normalized to [-1, 1]")
    return actions
