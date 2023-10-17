"""Base Language Model"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Type

from chatllm.llm_response import LLMResponse
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


class LLMRegister(object):
    """Class Decorater to register the list of supported models"""

    def __init__(self, llm_key):
        self.llm_key = llm_key
        self.llms = {}

    def __call__(self, clz: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        """Register the list of supported models"""
        logger.info(f"Adding LLM Provider = {self.llm_key} // {clz}")
        BaseLLMProvider.register_llm(self.llm_key, clz)
        return clz


class BaseLLMProvider(ABC):
    """Abstract base class for interfacing with language models."""

    llm_models: Dict[str, Any] = {}

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.llm_args = kwargs  # TODO: This is currently unused

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.model_name

    @staticmethod
    def register_llm(llm_key, llm_class) -> None:
        """Class Decorater to register the list of supported models"""
        BaseLLMProvider.llm_models[llm_key] = {
            "class": llm_class,
            "models": llm_class.get_supported_models(),
        }

    @staticmethod
    def registered_models() -> Dict[str, Any]:
        """Return a list of supported models."""
        return BaseLLMProvider.llm_models

    @abstractmethod
    async def load(self, **kwargs: Any) -> None:
        """Load the model."""

    @abstractmethod
    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:  # Generator[LLMResponse, Any, Any]:
        """Pass a single prompt value to the model and return model generations."""

    @abstractmethod
    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any | str, Any]:
        """Pass a single prompt value to the model and stream model generations."""

    async def generate_batch(
        self,
        prompt_values: List[PromptValue],
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        """Pass a sequence of prompt values to the model and return model generations."""
        responses = []
        for prompt_value in prompt_values:
            response = await self.generate(prompt_value, verbose=verbose, **kwargs)
            responses.append(response.get_first_sequence())
        return responses
