"""ADK tool boundary: candidate validation precedes every physical submission."""

from google.adk.tools import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from jsonschema import ValidationError

from roboweaver.contracts import ActionProposal


class ActionBridge:
    def __init__(self, runtime):
        self.runtime = runtime

    def tools(self):
        return [LongRunningFunctionTool(self.execute_action), self.finish_task, self.observe]

    async def execute_action(
        self, action_type: str, arguments: dict, observation_id: str, tool_context: ToolContext
    ) -> dict:
        """Submit one registered semantic action and wait for verified execution feedback.

        Select action_type and arguments from the current capability descriptions.
        observation_id must identify the observation used to make this decision.
        """
        runtime = self.runtime
        state = runtime.state
        call_id = tool_context.function_call_id
        if not call_id:
            raise ValueError("A tool call requires a stable call ID")
        if state.binding.get("function_call_id") == call_id:
            return {"status": "pending", "action_id": state.binding["action_id"]}
        binding = {
            "session_id": state.run_id,
            "invocation_id": tool_context.invocation_id,
            "function_call_id": call_id,
            "tool_name": "execute_action",
        }
        try:
            runtime.scheduler.append(
                state.run_id,
                state.plan_version,
                [
                    ActionProposal(
                        action_type=action_type, arguments=arguments, observation_id=observation_id
                    )
                ],
                binding=binding,
            )
        except (ValueError, ValidationError) as exc:
            with runtime.scheduler.ledger.transaction(state.run_id) as (db, current):
                current.binding = {**binding, "rejection": str(exc)}
            return {"status": "rejected", "reason": str(exc)}
        await runtime.scheduler.dispatch(state.run_id)
        return {"status": "pending", "action_id": runtime.state.binding["action_id"]}

    async def finish_task(self, tool_context: ToolContext) -> dict:
        """Explicitly close the action stream and independently verify the overall task goal."""
        runtime = self.runtime
        state = runtime.state
        runtime.scheduler.close(state.run_id, state.plan_version, state.next_sequence - 1)
        await runtime.refresh()
        result = runtime.verify("final")
        if result.verdict != "PASS":
            await runtime.recover()
        else:
            tool_context.actions.skip_summarization = True
        return {"status": runtime.state.status, "verification": result.model_dump(mode="json")}

    async def observe(self) -> dict:
        """Refresh observation when more evidence is needed; no physical action is submitted."""
        from roboweaver.context import context_data

        await self.runtime.refresh()
        return context_data(self.runtime.state)
