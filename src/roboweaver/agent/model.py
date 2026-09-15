"""Bound model I/O attempts while leaving the agent/tool loop inside ADK."""

import asyncio
from typing import Any

from google.adk.models.base_llm import BaseLlm
from pydantic import Field


class BudgetExceeded(RuntimeError):
    pass


class InvalidDecision(ValueError):
    def __init__(self, message, *, call_id=None, invocation_id=None):
        super().__init__(message)
        self.call_id, self.invocation_id = call_id, invocation_id


class RetryableModelError(ConnectionError):
    """Adapters map transient transport failures and rate limiting to this error."""


class BudgetedModel(BaseLlm):
    model: str = "roboweaver-budgeted"
    delegate: BaseLlm = Field(exclude=True)
    runtime: Any = Field(exclude=True)

    async def generate_content_async(self, llm_request, stream=False):
        while True:
            self.runtime.take_model_call()
            budget = self.runtime.state.task.budget
            received = False
            try:
                async with asyncio.timeout(budget.model_seconds):
                    async for response in self.delegate.generate_content_async(
                        llm_request, stream=stream
                    ):
                        received = True
                        yield response
                return
            except (TimeoutError, ConnectionError):
                with self.runtime.scheduler.ledger.transaction(self.runtime.run_id) as (db, state):
                    exhausted = received or state.api_retries >= budget.max_api_retries
                    if not exhausted:
                        state.api_retries += 1
                        self.runtime.scheduler.ledger.log(
                            db, state.run_id, "model_retry", {"count": state.api_retries}
                        )
                if exhausted:
                    raise
