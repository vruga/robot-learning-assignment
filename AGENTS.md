# Assignment repository

Use `uv` with the committed `uv.lock`. Keep ONNX Runtime telemetry disabled;
the package sets `ORT_DISABLE_TELEMETRY=1` before importing the runtime.

The environment/evaluator under `src/advaitic_eval/`, the pinned BiGym2
dependency, its assets/controller, and evaluation rules are fixed for a
distributed assignment release. Candidate changes belong in `submission/`
and method-specific dependency/configuration files. Suspected task bugs
must be reported and fixed in a new common release, not silently changed
for one candidate.

Required checks after evaluator changes:

```bash
uv run ruff check src submission tests
uv run ruff format --check src submission tests
ORT_DISABLE_TELEMETRY=1 MUJOCO_GL=egl uv run pytest -q
```

Never regenerate the upstream integrity manifest or the demonstration
endpoint fixture merely to get a failing test to pass. An intended change
requires a documented common-release update and revalidation.

Fixture pose injection is allowed only in evaluator tests. It must never
be presented as a policy rollout or a successful task-solving result.

Do not claim the 90% policy target was demonstrated by evaluator tests.
