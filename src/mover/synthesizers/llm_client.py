from typing import Any

import litellm
from litellm import CustomStreamWrapper, ModelResponse
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse


class LLMClient:
    """Configured LiteLLM calls, plus a text-only helper for synthesizers."""

    def __init__(
        self,
        model: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.params = dict(params or {})

    def complete(self, messages: list[dict[str, Any]]) -> ModelResponse | CustomStreamWrapper:
        """Return LiteLLM's Chat Completions response or stream unchanged."""
        return litellm.completion(model=self.model, messages=messages, **self.params)

    def responses(
        self, input: str | list[dict[str, Any]],
    ) -> ResponsesAPIResponse | BaseResponsesAPIStreamingIterator:
        """Return LiteLLM's Responses API result or stream unchanged."""
        return litellm.responses(model=self.model, input=input, **self.params)

    def create(self, messages: list[dict[str, Any]]) -> str:
        """Return the first text result; use complete() for metadata or streams."""
        if self.params.get("stream") or self.params.get("model_list") is not None:
            raise ValueError("create() returns one text response; use complete() for streams or batches")
        response = self.complete(messages)
        if (
            not isinstance(response, ModelResponse)
            or not response.choices
            or not isinstance(response.choices[0].message.content, str)
        ):
            raise ValueError("The model returned no text; use complete() to inspect the full response")
        return response.choices[0].message.content