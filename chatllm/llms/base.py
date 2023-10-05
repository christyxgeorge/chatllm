"""Base Language Model"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, List, Optional, Type


class LLMRegister(object):
    """Class Decorater to register the list of supported models"""

    def __init__(self, llm_key):
        self.llm_key = llm_key
        self.llms = {}

    def __call__(self, clz: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        """Register the list of supported models"""
        print(f"Adding LLM Provider = {self.llm_key} // {clz}")
        BaseLLMProvider.register_llm(self.llm_key, clz)
        return clz


class BaseLLMProvider(ABC):
    """Abstract base class for interfacing with language models."""

    llm_models = {}

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self.llm_args = kwargs

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.model_name

    @abstractmethod
    async def load(self, **kwargs: Any) -> None:
        """Load the model."""

    @abstractmethod
    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""

    @abstractmethod
    async def generate(
        self,
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> str:
        """Pass a single prompt value to the model and return model generations."""

    @staticmethod
    def register_llm(llm_key, llm_class) -> None:
        """Class Decorater to register the list of supported models"""
        BaseLLMProvider.llm_models[llm_key] = {
            "class": llm_class,
            "models": llm_class.get_supported_models(),
        }

    @staticmethod
    def registered_models() -> List[str]:
        """Return a list of supported models."""
        return BaseLLMProvider.llm_models

    async def generate_stream(
        self,
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""

    async def generate_batch(
        self,
        input_prompts: List[str],
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Pass a sequence of prompt values to the model and return model generations."""
        responses = []
        for prompt in input_prompts:
            response = await self.generate(prompt, verbose=verbose, **kwargs)
            responses.append(response["choices"][0]["text"])

        return responses

    # General Utility Functions [to format into the OpenAI response format]
    def get_response_template(self, num_prompt_tokens, usage=False):
        unique_id = f"cllm-{random.randint(10000000,99999999):08d}"
        usage = (
            {}
            if not usage
            else {
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": num_prompt_tokens,
                }
            }
        )
        result = {
            "id": unique_id,
            "object": "cllm.generation",
            "created": int(datetime.timestamp(datetime.now())),
            "model": self.model_name,
            **usage,
            "choices": [],
        }
        return result

    def format_choice(self, content: str | List[str], start_idx=0):
        choiceList = content if isinstance(choice, list) else [choice]
        return [
            {
                "message": {"role": "assistant", "content": c},
                "finish_reason": "stop",
                "index": i,
            }
            for i, c in enumerate(choiceList, start_idx)
        ]

    def format_delta(self, content: str | List[str], start_idx=0):
        contentList = content if isinstance(content, list) else [content]
        return [
            {
                "delta": {"role": "assistant", "content": c},
                "finish_reason": None,
                "index": i,
            }
            for i, c in enumerate(contentList, start_idx)
        ]

    def format_last_delta(self, start_idx=0, num_deltas=1):
        # TODO: What if we need n_deltas for multiple choices generated earlier....
        return [
            {"delta": {"role": "assistant", "content": ""}, "finish_reason": "stop", "index": i}
            for i in range(num_deltas)
        ]
