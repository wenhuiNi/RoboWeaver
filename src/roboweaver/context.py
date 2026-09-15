"""Build provider-neutral context without replacing authoritative domain state."""

from roboweaver.contracts import RunState


def context_data(state: RunState) -> dict:
    return {
        "task": state.task.model_dump(mode="json"),
        "observation": state.observation.model_dump(mode="json", exclude={"images"}),
        "capabilities": [c.model_dump(mode="json") for c in state.capabilities],
        "plan_version": state.plan_version,
        "recent_actions": [a.model_dump(mode="json") for a in state.actions[-8:]],
        "verification": [v.model_dump(mode="json") for v in state.verifications[-4:]],
        "remaining_model_calls": state.task.budget.max_model_calls - state.model_calls,
    }
