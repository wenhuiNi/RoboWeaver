"""Deterministic model responses for offline protocol checks, not model inference."""

import json
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
        self.requests.append(
            LlmRequest(
                contents=[c.model_copy(deep=True) for c in llm_request.contents],
                config=llm_request.config.model_copy(deep=True),
            )
        )
        if not self.responses:
            raise RuntimeError("Scripted model response budget exhausted")
        item = self.responses.pop(0)
        if "error" in item:
            if item["error"] == "rate_limit":
                from roboweaver.agent.model import RetryableModelError

                raise RetryableModelError("Synthetic rate limit")
            raise TimeoutError("Synthetic model timeout")
        context = {}
        for content in llm_request.contents:
            for part in content.parts or []:
                if part.text:
                    try:
                        parsed = json.loads(part.text)
                        if isinstance(parsed, dict) and "observation" in parsed:
                            context = parsed
                    except ValueError:
                        pass
                if part.function_response and "observation" in part.function_response.response:
                    context = part.function_response.response
        if "tool" in item:
            args = json.loads(
                json.dumps(item.get("arguments", {})).replace(
                    "$observation_id",
                    context.get("observation", {}).get("observation_id", "missing"),
                )
            )
            content = types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name=item["tool"], args=args))],
            )
        else:
            content = types.Content(role="model", parts=[types.Part(text=item.get("text", "Done"))])
        yield LlmResponse(content=content)
