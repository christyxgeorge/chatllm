"""Base Language Model"""
from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Type

from chatllm.llm_params import LLMConfig
from chatllm.llm_response import LLMResponse
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


class LLMRegister(object):
    """Class Decorater to register the list of supported models"""

    def __init__(self, *, config_class: Type[LLMConfig]) -> None:
        self.config_class = config_class

    def __call__(self, clz: Type[BaseLLMProvider]) -> Type[BaseLLMProvider]:
        """Register the list of supported models"""
        provider_key = clz.__module__.split(".")[-1]  # Use Module/Package Name (file name)
        logger.info(f"Adding LLM Provider = {provider_key} // {clz} // {self.config_class}")
        BaseLLMProvider.register_provider(provider_key, clz, self.config_class)
        return clz


class BaseLLMProvider(ABC):
    """Abstract base class for interfacing with language models."""

    llm_providers: Dict[str, Dict[str, Any]] = {}

    def __init__(self, model_name: str, model_cfg: LLMConfig, **kwargs: Any) -> None:
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.llm_args = kwargs  # TODO: This is currently unused

    @property
    def model(self) -> str:
        """Return the model name."""
        return self.model_name

    @staticmethod
    def register_provider(provider_key: str, provider_class, config_class: Type[LLMConfig]) -> None:
        BaseLLMProvider.llm_providers[provider_key] = {
            "class": provider_class,
            "config_class": config_class,
        }

    @classmethod
    def model_config(cls, model_config: Dict[str, Any]) -> LLMConfig:
        """Load the model config."""
        provider_key = model_config["provider"]
        config_class = BaseLLMProvider.llm_providers[provider_key]["config_class"]
        llm_config = config_class.create_config(model_config)
        return llm_config

    @staticmethod
    def provider_class(provider):
        provider_info = BaseLLMProvider.llm_providers[provider]
        return provider_info["class"]

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
