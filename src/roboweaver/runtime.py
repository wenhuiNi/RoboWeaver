"""An event-driven ADK session connected to a durable robot action scheduler."""

import asyncio
import base64
import json
import time
from functools import wraps
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from roboweaver.agent.bridge import ActionBridge
from roboweaver.agent.model import BudgetedModel, BudgetExceeded, InvalidDecision
from roboweaver.context import context_data
from roboweaver.contracts import TERMINAL, ActionStatus, TaskStatus, VerificationResult
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_verifier


def exclusive(method):
    @wraps(method)
    async def wrapped(self, *args, **kwargs):
        task = asyncio.current_task()
        if self._owner is task:
            return await method(self, *args, **kwargs)
        with self.scheduler.ledger.claim(self.run_id):
            self._owner = task
            try:
                return await method(self, *args, **kwargs)
            finally:
                self._owner = None

    return wrapped


class Runtime:
    def __init__(
        self, scheduler: Scheduler, run_id: str, model, session_path: Path, clock=time.time
    ):
        self.scheduler, self.run_id, self.model, self.clock = scheduler, run_id, model, clock
        self._owner = None
        self.session_path = Path(session_path).resolve()
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{self.session_path}")
        bridge = ActionBridge(self)
        agent = LlmAgent(
            name="robot_planner",
            model=BudgetedModel(delegate=model, runtime=self),
            tools=bridge.tools(),
            instruction="Use only registered actions. Call execute_action with the current observation ID. "
            "Wait for execution feedback. Use observe for fresh evidence. Call finish_task to request completion. "
            "Do not infer success from execution completion. The action catalog is in the context.",
            after_model_callback=self._after_model,
        )
        self.runner = Runner(
            app=App(
                name="roboweaver",
                root_agent=agent,
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            session_service=self.sessions,
        )

    @property
    def state(self):
        return self.scheduler.ledger.read(self.run_id)

    def take_model_call(self):
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            if state.frozen or state.status in {TaskStatus.CANCELED, TaskStatus.FAILED}:
                raise BudgetExceeded("Run no longer accepts model decisions")
            if state.model_calls >= state.task.budget.max_model_calls:
                raise BudgetExceeded("Model call budget exhausted")
            if self.clock() - state.created_at >= state.task.budget.total_seconds:
                raise BudgetExceeded("Run time budget exhausted")
            state.model_calls += 1
            self.scheduler.ledger.log(
                db,
                self.run_id,
                "model_call",
                {"count": state.model_calls, "model": self.model.model},
            )

    def _after_model(self, callback_context, llm_response):
        if self.state.frozen:
            raise BudgetExceeded("Run stopped while model was executing")
        calls = [
            p.function_call
            for p in (llm_response.content.parts if llm_response.content else [])
            if p.function_call
        ]
        if len(calls) > 1:
            raise InvalidDecision("Emit at most one action tool call per decision")

    async def refresh(self):
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            if state.observation_count >= state.task.budget.max_observations:
                raise BudgetExceeded("Observation budget exhausted")
            state.observation_count += 1
        async with asyncio.timeout(self.state.task.budget.io_seconds):
            observation = await self.scheduler.executor.observe()
        self.scheduler.observe(self.run_id, observation)
        return observation

    def verify(self, checkpoint):
        state = self.state
        result = VerificationResult.model_validate(
            load_verifier(state.task.verifier)(state.task, state.observation, checkpoint)
        )
        self.scheduler.verify(self.run_id, result)
        return result

    def _parts(self):
        parts = [types.Part(text=json.dumps(context_data(self.state)))]
        for image in self.state.observation.images:
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(image.data_base64, validate=True),
                    mime_type=image.mime_type,
                )
            )
        return parts

    @exclusive
    async def start(self):
        existing = await self.sessions.get_session(
            app_name="roboweaver", user_id="local", session_id=self.run_id
        )
        if existing is not None:
            raise ValueError("Run already has an ADK session; resume it instead")
        await self.sessions.create_session(
            app_name="roboweaver", user_id="local", session_id=self.run_id
        )
        return await self._run(types.Content(role="user", parts=self._parts()))

    async def _drive(self, message=None, invocation_id=None):
        paused = False
        async for event in self.runner.run_async(
            user_id="local",
            session_id=self.run_id,
            new_message=message,
            invocation_id=invocation_id,
        ):
            paused = (
                paused
                or bool(event.long_running_tool_ids)
                or any(
                    response.name == "execute_action"
                    and response.response.get("status") == "pending"
                    for response in event.get_function_responses()
                )
            )
            with self.scheduler.ledger.transaction(self.run_id) as (db, state):
                self.scheduler.ledger.log(
                    db,
                    self.run_id,
                    "agent_event",
                    {
                        "event_id": event.id,
                        "invocation_id": event.invocation_id,
                        "author": event.author,
                        "tool_calls": [f.name for f in event.get_function_calls()],
                    },
                )

        return paused

    async def _run(self, message=None, invocation_id=None):
        while True:
            try:
                remaining = self.state.task.budget.total_seconds - (
                    self.clock() - self.state.created_at
                )
                if remaining <= 0:
                    raise BudgetExceeded("Run time budget exhausted")
                async with asyncio.timeout(remaining):
                    paused = await self._drive(message, invocation_id)
                state = self.state
                if not paused and "action_id" in state.binding:
                    session = await self._session()
                    call_id = state.binding["function_call_id"]
                    paused = not any(
                        event.author == "user"
                        and any(r.id == call_id for r in event.get_function_responses())
                        for event in session.events
                    )
                if "rejection" in state.binding:
                    raise InvalidDecision(
                        state.binding["rejection"],
                        call_id=state.binding["function_call_id"],
                        invocation_id=state.binding["invocation_id"],
                    )
                if (
                    state.status
                    not in {
                        TaskStatus.SUCCEEDED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELED,
                        TaskStatus.BLOCKED,
                    }
                    and not state.closed
                    and not paused
                    and not any(a.status not in TERMINAL for a in state.actions)
                ):
                    raise InvalidDecision("Agent returned without an action or explicit completion")
                return state
            except InvalidDecision as exc:
                with self.scheduler.ledger.transaction(self.run_id) as (db, state):
                    state.invalid_proposals += 1
                    exhausted = state.invalid_proposals > state.task.budget.max_invalid_proposals
                    self.scheduler.ledger.log(
                        db,
                        self.run_id,
                        "invalid_decision",
                        {"reason": str(exc), "count": state.invalid_proposals},
                    )
                if exhausted:
                    await self.terminate("failure", "Invalid decision budget exhausted")
                    return self.state
                parts = self._parts()
                if exc.call_id:
                    parts.insert(
                        0,
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=exc.call_id,
                                name="execute_action",
                                response={"status": "rejected", "reason": str(exc)},
                            )
                        ),
                    )
                else:
                    parts.insert(0, types.Part(text="Invalid decision: " + str(exc)))
                with self.scheduler.ledger.transaction(self.run_id) as (db, state):
                    if "rejection" in state.binding:
                        state.binding = {}
                message = types.Content(role="user", parts=parts)
                invocation_id = exc.invocation_id
            except (ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
                await self.terminate("failure", type(exc).__name__)
                return self.state

    async def terminate(self, reason, detail):
        if self.state.stop_intent in {"cancel", "timeout"}:
            reason = self.state.stop_intent
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            state.reason = detail
            self.scheduler.ledger.log(
                db, self.run_id, "termination_requested", {"reason": reason, "detail": detail}
            )
        await self.scheduler.request_stop(self.run_id, reason)
        await self.scheduler.tick(self.run_id)

    async def cancel(self):
        if self.state.status not in {TaskStatus.SUCCEEDED, TaskStatus.CANCELED, TaskStatus.FAILED}:
            await self.terminate("cancel", "Operator canceled")
        return self.state

    async def recover(self):
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            exhausted = state.replans >= state.task.budget.max_replans
            if not exhausted:
                state.replans += 1
        if exhausted:
            await self.terminate("failure", "Replanning budget exhausted")
            return False
        await self.scheduler.request_stop(self.run_id, "replan")
        await self.scheduler.tick(self.run_id)
        if any(a.status not in TERMINAL for a in self.state.actions):
            return False
        await self.refresh()
        self.scheduler.revise(self.run_id, self.state.plan_version, self.state.observation)
        return True

    @exclusive
    async def feedback(self, event):
        self.scheduler.apply(event)
        return await self.continue_execution()

    async def _session(self):
        return await self.sessions.get_session(
            app_name="roboweaver", user_id="local", session_id=self.run_id
        )

    @exclusive
    async def resume(self):
        state = await self.scheduler.tick(self.run_id)
        if state.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELED}:
            return state
        session = await self._session()
        if session is None:
            if state.actions:
                raise ValueError(
                    "ADK session missing for an existing execution; reconciliation required"
                )
            return await self.start()
        if state.pending_feedback:
            return await self._deliver_feedback()
        binding = state.binding
        if binding:
            # Crash inside the tool may leave a persisted call without its pending response.
            has_pending = any(
                event.author != "user"
                and any(r.id == binding["function_call_id"] for r in event.get_function_responses())
                for event in session.events
            )
            if not has_pending:
                if not any(
                    any(
                        call.id == binding["function_call_id"]
                        for call in event.get_function_calls()
                    )
                    for event in session.events
                ):
                    raise ValueError("Persisted tool binding has no matching ADK call")
                # The tool's durable intent already exists. Restore its receipt without executing it again.
                receipt = types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=binding["function_call_id"],
                                name=binding["tool_name"],
                                response={
                                    "status": "pending",
                                    "action_id": binding.get("action_id"),
                                },
                            )
                        )
                    ],
                )
                await self.sessions.append_event(
                    session=session,
                    event=Event(
                        author="robot_planner",
                        invocation_id=binding["invocation_id"],
                        content=receipt,
                    ),
                )
            return await self.continue_execution()
        if session.events:
            return await self._run(invocation_id=session.events[-1].invocation_id)
        return await self._run(types.Content(role="user", parts=self._parts()))

    async def _deliver_feedback(self):
        pending = self.state.pending_feedback
        if pending is None:
            return self.state
        session = await self._session()
        if session is None:
            raise ValueError("ADK session missing for pending feedback")
        delivered = any(
            event.author == "user"
            and any(r.id == pending["call_id"] for r in event.get_function_responses())
            for event in session.events
        )
        message = None if delivered else types.Content.model_validate(pending["message"])
        result = await self._run(message, pending["invocation_id"])
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            if state.pending_feedback and state.pending_feedback["call_id"] == pending["call_id"]:
                state.pending_feedback = None
                self.scheduler.ledger.log(
                    db, self.run_id, "feedback_delivered", {"call_id": pending["call_id"]}
                )
        if self.state.binding.get("function_call_id") != pending["call_id"]:
            return await self.continue_execution()
        return result

    @exclusive
    async def continue_execution(self):
        state = self.state
        if state.status in {
            TaskStatus.FAILED,
            TaskStatus.CANCELED,
            TaskStatus.SUCCEEDED,
        } or state.stop_intent in {"cancel", "timeout", "failure"}:
            return state
        if state.pending_feedback:
            return await self._deliver_feedback()
        if not state.binding or "action_id" not in state.binding:
            return state
        action = next(a for a in state.actions if a.action_id == state.binding["action_id"])
        if action.status not in TERMINAL:
            return state
        try:
            await self.refresh()
            previous = next(
                (v for v in reversed(self.state.verifications) if v.checkpoint == action.action_id),
                None,
            )
            result = previous
            if action.status == ActionStatus.COMPLETED and not action.verified:
                result = self.verify(action.action_id)
            # A crash after revision must not consume another replan for the old action.
            if action.plan_version == self.state.plan_version and (
                action.status != ActionStatus.COMPLETED or (result and result.verdict != "PASS")
            ):
                if not await self.recover():
                    return self.state
            binding = self.state.binding.copy()
            response = types.Part(
                function_response=types.FunctionResponse(
                    id=binding["function_call_id"],
                    name=binding["tool_name"],
                    response={
                        "status": action.status,
                        "verification": result.model_dump(mode="json") if result else None,
                    },
                )
            )
            message = types.Content(role="user", parts=[response, *self._parts()])
            with self.scheduler.ledger.transaction(self.run_id) as (db, state):
                state.pending_feedback = {
                    "call_id": binding["function_call_id"],
                    "invocation_id": binding["invocation_id"],
                    "message": message.model_dump(mode="json"),
                }
                self.scheduler.ledger.log(
                    db, self.run_id, "feedback_prepared", {"call_id": binding["function_call_id"]}
                )
            return await self._deliver_feedback()
        except (ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
            await self.terminate("failure", type(exc).__name__)
            return self.state

    async def aclose(self):
        await self.sessions.close()
