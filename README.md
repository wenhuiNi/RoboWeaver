# RoboWeaver

A harness for long-horizon embodied tasks with extensible atomic action streams and VLA execution.

RoboWeaver connects vision-language models (VLMs) to robot execution through a feedback loop: observe, decide, execute, verify, and replan. Tasks define the goal; execution adapters define the available actions.

> **Status:** Early development. A runnable package and validated robot integrations are not yet available.

## Design

```text
Task + observations → VLM agent → Atomic action stream → Robot / VLA runtime
                          ↑                                      │
                          └──── Verification + feedback ─────────┘
```

The planned implementation uses Google ADK for the upper-level agent, with separate interfaces for model APIs and robot execution.

- **Extensible actions** — action types and parameters come from execution capabilities.
- **User-defined tasks** — goals, constraints, and verification criteria are independent of the agent loop.
- **Feedback-driven execution** — execution results and new observations guide subsequent actions.
- **Replaceable adapters** — model providers and robot transports stay outside the core task contract.

An atomic action is an independently scheduled and tracked execution unit. Its granularity is defined by the execution adapter; continuous control remains in the robot runtime.
