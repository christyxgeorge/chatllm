"""Language Model To Interface With OpenAI's API"""
from __future__ import annotations

import logging
import os

from typing import Any, AsyncGenerator, Dict, List, Tuple, Union

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

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


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


@LLMRegister(config_class=OpenAIConfig)
class OpenAIChat(BaseLLMProvider):
    """Abstract base class for interfacing with OpenAI language models."""

    def __init__(self, model_name: str, model_cfg: LLMConfig, **kwargs) -> None:
        super().__init__(model_name, model_cfg, **kwargs)
        self.client = (
            openai.ChatCompletion if self.model_cfg.mtype.is_chat_model else openai.Completion
        )
        self.encoding = tiktoken.encoding_for_model(model_name)
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[str]:
        """Return a list of supported models."""
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        model_list = openai.Model.list()
        model_list = [m["id"] for m in model_list["data"]]
        logger.debug(f"Open AI Model List = {len(model_list)} // {model_list}")
        model_list = [m for m in model_list if m.startswith("gpt")]
        return model_list

    async def load(self, **kwargs: Any) -> None:
        """Load the model. Nothing to do in the case of OpenAI"""
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.encoding.encode(prompt)
        # print(f"Encoding = {tokens}")
        return len(tokens)

    def format_prompt(
        self, prompt_value: PromptValue
    ) -> Tuple[Union[List[dict[str, str]], str], int]:
        """Format the prompt for OpenAI"""
        formatted_prompt = (
            prompt_value.to_messages()
            if self.model_cfg.mtype.is_chat_model
            else prompt_value.to_string()
        )
        return formatted_prompt, self.get_token_count(prompt_value.to_string())  # type: ignore

    def validate_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Validate the kwargs for the model"""
        # Rename 'num_sequences' to 'n', repeat_penalty to freq_penalty
        kwargs["n"] = kwargs.pop("num_sequences", 1)
        kwargs["frequency_penalty"] = kwargs.pop("repeat_penalty", 0.0)
        return kwargs

    def get_openai_args(self, openai_prompt, streaming: bool, **kwargs):
        """Get the arguments for the LLM call"""
        openai_kwargs = {
            "model": self.model_name,
            "stream": streaming,
        }
        openai_kwargs.update(kwargs)
        if self.model_cfg.mtype.is_chat_model:
            openai_kwargs["messages"] = openai_prompt
        else:
            openai_kwargs["prompt"] = openai_prompt
        return openai_kwargs

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
        openai_kwargs = self.get_openai_args(openai_prompt, False, **validated_kwargs)
        # TODO: Need to support tenacity to retry errors!
        response = await self.client.acreate(**openai_kwargs)
        llm_response.set_openai_response(
            response, chat_completion=self.model_cfg.mtype.is_chat_model
        )
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
        openai_kwargs = self.get_openai_args(openai_prompt, True, **validated_kwargs)
        stream = await self.client.acreate(**openai_kwargs)

        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            async for response_delta in stream:
                if self.model_cfg.mtype.is_chat_model:
                    llm_response.add_openai_delta(response_delta)
                else:
                    text_delta = [c["text"] for c in response_delta["choices"]]
                    llm_response.add_delta(text_delta)
                yield llm_response

        return async_generator()
