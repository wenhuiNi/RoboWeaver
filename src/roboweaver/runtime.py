"""An event-driven ADK session connected to a durable robot action scheduler."""

import base64
import json
import time
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from roboweaver.agent.bridge import ActionBridge
from roboweaver.context import context_data
from roboweaver.contracts import TERMINAL, ActionStatus, TaskStatus, VerificationResult
from roboweaver.scheduler import Scheduler
from roboweaver.tasks import load_verifier


class Runtime:
    def __init__(
        self, scheduler: Scheduler, run_id: str, model, session_path: Path, clock=time.time
    ):
        self.scheduler, self.run_id, self.model, self.clock = scheduler, run_id, model, clock
        self.session_path = Path(session_path).resolve()
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{self.session_path}")
        bridge = ActionBridge(self)
        agent = LlmAgent(
            name="robot_planner",
            model=model,
            tools=bridge.tools(),
            instruction="Use only registered actions. Call execute_action with the current observation ID. "
            "Wait for execution feedback. Use observe for fresh evidence. Call finish_task to request completion. "
            "Do not infer success from execution completion. The action catalog is in the context.",
            before_model_callback=self._before_model,
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

    def _before_model(self, callback_context, llm_request):
        with self.scheduler.ledger.transaction(self.run_id) as (db, state):
            if state.model_calls >= state.task.budget.max_model_calls:
                raise ValueError("Model call budget exhausted")
            state.model_calls += 1
            self.scheduler.ledger.log(
                db,
                self.run_id,
                "model_call",
                {"count": state.model_calls, "model": self.model.model},
            )

    async def refresh(self):
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

    async def _run(self, message=None, invocation_id=None):
        try:
            async for event in self.runner.run_async(
                user_id="local",
                session_id=self.run_id,
                new_message=message,
                invocation_id=invocation_id,
            ):
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
            state = self.state
            active = any(a.status not in TERMINAL for a in state.actions)
            if not active and not state.closed and state.status == TaskStatus.READY:
                raise ValueError("Agent returned without an action or explicit stream completion")
        except (ValueError, RuntimeError, TimeoutError, ConnectionError) as exc:
            with self.scheduler.ledger.transaction(self.run_id) as (db, state):
                state.reason = type(exc).__name__
                state.status = TaskStatus.FAILED
                self.scheduler.ledger.log(
                    db, self.run_id, "agent_error", {"type": type(exc).__name__}
                )
            raise
        return self.state

    async def feedback(self, event):
        self.scheduler.apply(event)
        return await self.continue_execution()

    async def continue_execution(self):
        state = self.state
        if not state.binding:
            return state
        action = next(a for a in state.actions if a.action_id == state.binding["action_id"])
        if action.status not in TERMINAL:
            return state
        await self.refresh()
        if action.status == ActionStatus.COMPLETED and not action.verified:
            result = self.verify(action.action_id)
        else:
            result = None
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
        return await self._run(
            types.Content(role="user", parts=[response, *self._parts()]), binding["invocation_id"]
        )

    async def aclose(self):
        await self.sessions.close()
