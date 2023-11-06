"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, List, Tuple, cast

import google.generativeai as palm

from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.llms.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMParam,
    MaxTokens,
    NumSequences,
    RepeatPenalty,
    Temperature,
    TopK,
    TopP,
)
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


class Palm2Config(LLMConfig):
    # https://developers.generativeai.google/api/python/google/generativeai
    # Handle parameter variations
    max_tokens: LLMParam = MaxTokens(
        name="max_output_tokens", min=0, max=1024, step=64, default=128
    )
    temperature: LLMParam = Temperature(min=0, max=1, default=0.75, step=0.05)
    length_penalty: LLMParam = LengthPenalty(active=False)
    repeat_penalty: LLMParam = RepeatPenalty(active=False)
    num_sequences: LLMParam = NumSequences(
        name="candidate_count", min=1, max=8, default=1
    )
    top_k: LLMParam = TopK(min=0, max=40, default=40, step=10)


PALM2_MODEL_LIST: List[Palm2Config] = [
    Palm2Config(name="chat-bison", desc="PaLM-2 for Chat", ctx=8192, cpt=0.0),
    Palm2Config(
        name="chat-bison-32k",
        desc="PaLM-2 for Chat (32K)",
        ctx=32768,
        cpt=0.0,
    ),
    Palm2Config(
        name="text-bison",
        desc="PaLM-2 for Chat",
        ctx=8192,
        cpt=0.0,
    ),
    Palm2Config(
        name="code-bison",
        desc="Codey for code generation",
        top_p=TopP(active=False),
        top_k=TopK(active=False),
        ctx=6144,
        cpt=0.0,
    ),
    Palm2Config(
        name="codechat-bison",
        desc="Codey for code chat",
        ctx=6144,
        cpt=0.0,
    ),
]


@LLMRegister("vertexai")
class Palm2API(BaseLLMProvider):
    """Class for interfacing with GCP PaLM2 models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        palm.configure(api_key=os.environ["PALM2_API_KEY"])
        self.llm = palm

    @staticmethod
    def get_supported_models() -> List[LLMConfig]:
        """Return a list of supported models."""
        return cast(List[LLMConfig], PALM2_MODEL_LIST)

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens: List[str] = []
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple[str, int]:
        """Format the prompt for Vertex AI Predictions: Nothing to be done!"""
        formatted_prompt = prompt_value.to_string()
        return formatted_prompt, self.get_token_count(formatted_prompt)

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs passed to the model"""
        # Rename max_tokens to max output tokens
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens", 128)
        kwargs["candidate_count"] = kwargs.pop("num_sequences", 1)
        return kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Palm2 API has an async API only for chat!.
        """
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)

        prediction = self.llm.generate_text(formatted_prompt, **validated_kwargs)
        llm_response.set_response(prediction.text, ["stop"])
        llm_response.set_token_count(num_tokens, 0)
        return llm_response

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any | str, Any]:
        """
        Pass a single prompt value to the model and stream model generations.
        Note: PalM2 API does not have a streaming API.
        """
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        # In case of streaming, candidate count is not used as it is always one
        validated_kwargs.pop("candidate_count")

        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
        prediction = self.llm.generate_text(formatted_prompt, **validated_kwargs)

        logger.info(
            f"Vertex.AI prediction using {self.model_name}; Args = {validated_kwargs}"
        )

        # Wrap it in an async_generator!
        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            async for text_chunk in prediction:
                llm_response.add_delta(text_chunk.text)
                yield llm_response

            # Last token!
            if llm_response.completion_tokens >= kwargs.get("max_tokens", 0):
                finish_reason = "length"
            else:
                finish_reason = "stop"
            llm_response.add_last_delta(finish_reasons=[finish_reason])
            yield llm_response

        return async_generator()
