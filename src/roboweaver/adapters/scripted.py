"""Deterministic model responses for offline protocol checks, not model inference."""

from collections.abc import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field


class ScriptedModel(BaseLlm):
    model: str = "roboweaver-scripted"
    responses: list[dict] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list, exclude=True)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request.model_copy(deep=True))
        if not self.responses:
            raise RuntimeError("Scripted model response budget exhausted")
        yield LlmResponse(content=types.Content.model_validate(self.responses.pop(0)))
