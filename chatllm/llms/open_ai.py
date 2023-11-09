"""Language Model To Interface With OpenAI's API"""
from __future__ import annotations

import os

from typing import Any, AsyncGenerator, List, Tuple

import openai
import tiktoken

from chatllm.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMParam,
    NumSequences,
    RepeatPenalty,
    Temperature,
    TopK,
)
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompts import PromptValue


class OpenAIConfig(LLMConfig):
    # Handle parameter variations
    temperature: LLMParam = Temperature(min=0, max=2)
    num_sequences: LLMParam = NumSequences(name="n")
    repeat_penalty: LLMParam = RepeatPenalty(
        min=-2.0,
        max=-2.0,
        default=0.0,
        name="frequency_penalty",
        label="Frequency Penalty",
        desc="Positive values penalize new tokens based on their existing frequency in the text so far",
    )
    # Unsupported Parameters
    top_k: LLMParam = TopK(active=False)
    length_penalty: LLMParam = LengthPenalty(active=False)


@LLMRegister()
class OpenAIChat(BaseLLMProvider):
    """Abstract base class for interfacing with OpenAI language models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.client = openai.ChatCompletion
        self.encoding = tiktoken.encoding_for_model(model_name)
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[LLMConfig]:
        """Return a list of supported models."""
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        # model_list = openai.Model.list()
        # model_list = [m["id"] for m in model_list["data"]]
        # print(f"Open AI Model List = {len(model_list)} // {model_list}")
        # model_list = [m for m in model_list if m.startswith("gpt")]
        model_list: List[LLMConfig] = [
            OpenAIConfig(
                name="gpt-3.5-turbo", key="g35", desc="OpenAI GPT-3.5 Turbo", ctx=4096, cpt=0.01
            ),
            OpenAIConfig(
                name="gpt-3.5-turbo-16k",
                key="g35-16k",
                desc="OpenAI GPT-3.5 Turbo",
                ctx=16384,
                cpt=0.01,
            ),
            OpenAIConfig(name="gpt-4", key="g4", desc="OpenAI GPT-4", ctx=4096, cpt=0.02),
            OpenAIConfig(
                name="gpt-3.5-turbo-instruct",
                key="dv",
                desc="OpenAI GPT-3.5 Turbo Instruct / Davinci",
                ctx=4096,
                cpt=0.005,
            ),
        ]
        # return ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-3.5-turbo-instruct"]
        return model_list

    async def load(self, **kwargs: Any) -> None:
        """Load the model. Nothing to do in the case of OpenAI"""
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.encoding.encode(prompt)
        # print(f"Encoding = {tokens}")
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple[list[dict[str, str]], int]:
        """Format the prompt for OpenAI"""
        formatted_prompt = prompt_value.to_messages()
        return formatted_prompt, self.get_token_count(prompt_value.to_string())

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs for the model"""
        # Rename 'num_sequences' to 'n', repeat_penalty to freq_penalty
        kwargs["n"] = kwargs.pop("num_sequences", 1)
        kwargs["frequency_penalty"] = kwargs.pop("repeat_penalty", 0.0)
        return kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        openai_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)

        # TODO: Need to support tenacity to retry errors!
        response = await self.client.acreate(
            model=self.model_name,
            messages=openai_prompt,
            stream=False,
            **validated_kwargs,
        )
        llm_response.set_openai_response(response)
        return llm_response

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any | str, Any]:
        """Pass a single prompt value to the model and stream model generations."""
        openai_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        num_sequences = kwargs.get("n", 1)
        llm_response = LLMResponse(
            model=self.model, num_sequences=num_sequences, prompt_tokens=num_tokens
        )

        stream = await self.client.acreate(
            model=self.model_name,
            messages=openai_prompt,
            stream=True,
            **validated_kwargs,
        )

        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            async for response_delta in stream:
                llm_response.add_openai_delta(response_delta)
                yield llm_response

        return async_generator()
