"""Language Model To Interface With OpenAI's API"""
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, List, Optional

import openai
import tiktoken
from chatllm.llms.base import BaseLLMProvider, LLMRegister


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
        # print(f"API Key = {openai.api_key} // {os.environ.get('OPENAI_API_KEY')}")
        model_list = openai.Model.list()
        model_list = [m["id"] for m in model_list["data"]]
        # print(f"Open AI Model List = {len(model_list)} // {model_list}")
        model_list = [m for m in model_list if m.startswith("gpt")]
        return ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-3.5-turbo-instruct"]

    async def load(self, **kwargs: Any) -> None:
        """Load the model. Nothing to do in the case of OpenAI"""
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.encoding.encode(prompt)
        # print(f"Encoding = {tokens}")
        return len(tokens)

    def format_prompt(self, prompt: str) -> List[Any]:
        """Format the prompt for OpenAI"""
        return [{"role": "user", "content": prompt}]

    async def generate(
        self,
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        kwargs.setdefault("temperature", 0.9)
        kwargs.setdefault("max_tokens", 100)
        kwargs.setdefault("top_p", 1)
        kwargs.setdefault("frequency_penalty", 0.0)
        kwargs.setdefault("presence_penalty", 0.6)

        openai_prompt = self.format_prompt(input_prompt)
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
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        kwargs.setdefault("temperature", 0.9)
        kwargs.setdefault("max_tokens", 100)
        kwargs.setdefault("top_p", 1)
        kwargs.setdefault("frequency_penalty", 0.0)
        kwargs.setdefault("presence_penalty", 0.6)

        openai_prompt = self.format_prompt(input_prompt)
        result = await self.client.acreate(
            model=self.model_name,
            messages=openai_prompt,
            stream=True,
            **kwargs,
        )
        return result
