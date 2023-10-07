"""Language Model To Interface With OpenAI's API"""
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, List, Optional

import openai
import tiktoken
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompt import PromptValue


@LLMRegister("openai")
class OpenAIChat(BaseLLMProvider):
    """Abstract base class for interfacing with language models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.client = openai.ChatCompletion
        self.encoding = tiktoken.encoding_for_model(model_name)
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return a list of supported models."""
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        model_list = openai.Model.list()
        model_list = [m["id"] for m in model_list["data"]]
        # print(f"Open AI Model List = {len(model_list)} // {model_list}")
        model_list = [m for m in model_list if m.startswith("gpt")]
        return ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-3.5-turbo-instruct"]

    async def load(self, **kwargs: Any) -> None:
        """Load the model. Nothing to do in the case of OpenAI"""
        pass

    def get_params(self) -> List[str]:
        """Return Parameters supported by the model"""
        return {
            "max_tokens": {"minimum": 0, "maximum": 4096, "default": 128, "step": 64},
            "temperature": {"minimum": 0, "maximum": 2, "default": 1, "step": 0.1},
            "top_p": {"minimum": 0, "maximum": 1, "default": 1, "step": 0.1},
        }

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.encoding.encode(prompt)
        # print(f"Encoding = {tokens}")
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> List[dict]:
        """Format the prompt for OpenAI"""
        formatted_prompt = prompt_value.to_messages()
        return formatted_prompt, self.get_token_count(prompt_value.to_string())

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        # kwargs.setdefault("frequency_penalty", 0.0)
        # kwargs.setdefault("presence_penalty", 0.6)
        openai_prompt, num_tokens = self.format_prompt(prompt_value)
        # TODO: Need to support tenacity to retry errors!
        result = await self.client.acreate(
            model=self.model_name,
            messages=openai_prompt,
            stream=False,
            **kwargs,
        )
        return result

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        # TODO: Handle these parameters in the UI!
        # kwargs.setdefault("frequency_penalty", 0.0)
        # kwargs.setdefault("presence_penalty", 0.6)

        openai_prompt, num_tokens = self.format_prompt(prompt_value)
        result = await self.client.acreate(
            model=self.model_name,
            messages=openai_prompt,
            stream=True,
            **kwargs,
        )
        return result
