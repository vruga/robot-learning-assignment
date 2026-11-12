"""Balanced, deterministic evaluation plans; no simulator dependency."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

PROTOCOL_VERSION = "advaitic-plate-v1"
UPSTREAM_COMMIT = "d71501359b215b9d78e27dcb70b220412d474dfa"
SLOT_COUNT = 6
MIN_EPISODES = 30
PUBLIC_SEED_START = 7_310_000


@dataclass(frozen=True)
class Episode:
    seed: int
    source_slot: int  # zero-based; not handed to the policy


def source_slot_for_seed(seed: int) -> int:
    """Mirror the pinned move_plate reset's two rack draws and slot choice.

    The actual reset slot is also checked in the simulator, before running
    the policy. A future upstream change must fail rather than silently
    invalidate balanced coverage. We do not change the upstream reset.
    """
    rng = np.random.RandomState(seed)
    bounds = np.array([0.05, 0.05, 0.0])
    rng.uniform(-bounds, bounds)
    rng.uniform(-bounds, bounds)
    return int(rng.choice(np.arange(SLOT_COUNT), size=1, replace=False)[0])


def make_plan(episodes: int = MIN_EPISODES, seed_start: int = PUBLIC_SEED_START) -> list[Episode]:
    if episodes < MIN_EPISODES or episodes % SLOT_COUNT:
        raise ValueError(
            "Scored runs require >=30 episodes, in multiples of 6 (30, 36, 42, 60...)."
        )
    if not 0 <= seed_start < 2**32 - episodes * 100:
        raise ValueError("seed_start is outside the supported uint32 range")
    remaining = [episodes // SLOT_COUNT] * SLOT_COUNT
    plan = []
    seed = seed_start
    while len(plan) < episodes:
        slot = source_slot_for_seed(seed)
        if remaining[slot]:
            plan.append(Episode(seed, slot))
            remaining[slot] -= 1
        seed += 1
        if seed - seed_start > episodes * 100:
            raise RuntimeError("Could not construct balanced coverage")
    return plan


def plan_digest(plan: list[Episode]) -> str:
    raw = json.dumps([asdict(ep) for ep in plan], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def summarize(records: list[dict], planned: int, *, scored: bool, integrity_ok: bool) -> dict:
    successes = sum(bool(r["success"]) for r in records)
    count = len(records)
    coverage = Counter(r["source_slot"] for r in records)
    balanced = (
        count == planned
        and planned >= MIN_EPISODES
        and planned % SLOT_COUNT == 0
        and all(coverage[s] == planned // SLOT_COUNT for s in range(SLOT_COUNT))
    )
    valid = scored and integrity_ok and balanced and all(r.get("completed", False) for r in records)
    # Integer arithmetic avoids rounding 89.999... up to a pass.
    passed = valid and successes * 10 >= planned * 9
    return {
        "protocol": PROTOCOL_VERSION,
        "planned_episodes": planned,
        "completed_episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "required_successes": (9 * planned + 9) // 10,
        "integrity_ok": integrity_ok,
        "balanced_slot_coverage": balanced,
        "qualifying_run": valid,
        "meets_success_criterion": passed,
        "status": "PASS" if passed else ("FAIL" if valid else "NON_QUALIFYING"),
        "per_source_slot": {
            str(s + 1): {
                "episodes": coverage[s],
                "successes": sum(bool(r["success"]) for r in records if r["source_slot"] == s),
            }
            for s in range(SLOT_COUNT)
        },
    }
