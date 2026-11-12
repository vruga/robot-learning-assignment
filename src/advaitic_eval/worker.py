"""Candidate side of the policy interface; stdout is reserved for protocol data."""

from __future__ import annotations

import contextlib
import importlib
import json
import sys
import traceback

from .wire import unpack_array


def main():
    spec = sys.argv[1]
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("Policy must be module:factory")
    protocol_out = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        policy = getattr(importlib.import_module(module), name)()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "reset":
                    if hasattr(policy, "reset"):
                        policy.reset()
                    response = {"ok": True}
                elif request["op"] == "act":
                    obs = {key: unpack_array(value) for key, value in request["obs"].items()}
                    result = policy(obs)
                    response = {"actions": result.tolist() if hasattr(result, "tolist") else result}
                else:
                    raise ValueError("Unknown operation")
            protocol_out.write(json.dumps(response, allow_nan=False) + "\n")
            protocol_out.flush()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            protocol_out.write(json.dumps({"error": "policy_error"}) + "\n")
            protocol_out.flush()


if __name__ == "__main__":
    main()
