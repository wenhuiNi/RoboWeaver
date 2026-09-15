# RoboWeaver

A harness for long-horizon embodied tasks with extensible atomic action streams and VLA execution.

RoboWeaver connects vision-language models (VLMs) to robot execution through a feedback loop: observe, decide, execute, verify, and replan. Tasks define the goal; execution adapters define the available actions.

> **Status:** Offline prototype. The ADK agent loop runs with a scripted model and synthetic executor. Real model APIs and robot integrations are not yet validated.

## Design

```text
Task + observations → VLM agent → Atomic action stream → Robot / VLA runtime
                          ↑                                      │
                          └──── Verification + feedback ─────────┘
```

The implementation uses Google ADK for the upper-level agent, with separate interfaces for model APIs and robot execution.

- **Extensible actions** — action types and parameters come from execution capabilities.
- **User-defined tasks** — goals, constraints, and verification criteria are independent of the agent loop.
- **Feedback-driven execution** — execution results and new observations guide subsequent actions.
- **Replaceable adapters** — model providers and robot transports stay outside the core task contract.

An atomic action is an independently scheduled and tracked execution unit. Its granularity is defined by the execution adapter; continuous control remains in the robot runtime.

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run roboweaver --help
```

Run the bundled synthetic scenario:

```bash
uv run roboweaver run --mode mock \
  --task tests/fixtures/task.json \
  --capabilities tests/fixtures/capabilities.json \
  --scenario tests/fixtures/scenario.json \
  --workdir .local/demo
uv run roboweaver inspect --workdir .local/demo
```

Use a fresh working directory for each run. Results and execution events are written there.
The scenario checks orchestration; its scripted actions do not demonstrate robot or VLA performance.

## Tests

```bash
uv run pytest tests/smoke -m smoke -q
uv run pytest tests/integration -m integration -q
```

Tests run offline by default. Local environments, caches, and test results are excluded from Git.
