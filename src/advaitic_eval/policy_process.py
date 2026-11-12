"""Keep candidate imports out of the process owning the trusted simulator."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

from .wire import pack_array


class PolicyProcess:
    def __init__(
        self,
        spec: str,
        root: Path,
        log: Path,
        *,
        timeout: float = 120.0,
        python: str = sys.executable,
    ):
        self.timeout = timeout
        self.log = log.open("ab")
        env = os.environ.copy()
        # Worker package comes from this evaluator; only candidate imports use root.
        package_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join([package_root, str(root.resolve())])
        try:
            self.process = subprocess.Popen(
                [python, "-m", "advaitic_eval.worker", spec],
                cwd=root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                env=env,
            )
        except Exception:
            self.log.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.failed = False

    def _exchange(self, request):
        if self.failed or self.process.poll() is not None:
            raise RuntimeError("Policy process is unavailable")
        payload = memoryview((json.dumps(request, allow_nan=False) + "\n").encode())
        response = bytearray()
        deadline = time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Policy did not respond before the per-call timeout")
                    events = selector.select(remaining)
                    for key, _ in events:
                        if key.fileobj is self.process.stdin:
                            try:
                                written = os.write(key.fd, payload[:65536])
                            except BlockingIOError:
                                continue
                            payload = payload[written:]
                            if not payload:
                                selector.unregister(self.process.stdin)
                        else:
                            try:
                                chunk = os.read(key.fd, 65536)
                            except BlockingIOError:
                                continue
                            if not chunk:
                                raise RuntimeError("Policy exited before responding")
                            response.extend(chunk)
                            if len(response) > 1_000_000:
                                raise RuntimeError("Policy returned an oversized response")
                            if b"\n" in response:
                                if (
                                    payload
                                    or response.count(b"\n") != 1
                                    or not response.endswith(b"\n")
                                ):
                                    raise RuntimeError("Unexpected policy protocol response")
                                result = json.loads(response)
                                if not isinstance(result, dict):
                                    raise RuntimeError("Policy response must be an object")
                                return result
        except Exception:
            # A late response must never be mistaken for the next episode's reset.
            self.failed = True
            if self.process.poll() is None:
                self.process.kill()
            raise

    def request(self, request):
        result = self._exchange(request)
        if "error" in result:
            raise RuntimeError("Policy raised an error; inspect policy.stderr.log")
        return result

    def reset(self):
        self.request({"op": "reset"})

    def act(self, obs):
        return self.request({"op": "act", "obs": {k: pack_array(v) for k, v in obs.items()}})[
            "actions"
        ]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
