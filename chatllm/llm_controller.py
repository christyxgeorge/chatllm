"""Interface from Gradio/CLI to the LLM models."""
import json
import logging
import time
from typing import Any, AsyncGenerator, List

from chatllm.llms.base import BaseLLMProvider
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai:gpt-3.5-turbo"


class LLMController:
    """A chat Controller for conversational models."""

    def __init__(self) -> None:
        self.model_name = None
        self.llm = None
        self.model_map = BaseLLMProvider.registered_models()

    def get_model_list(self) -> List[str]:
        """return the list of models"""
        models = [
            f"{llm_key}:{m}"
            for llm_key, llm_info in self.model_map.items()
            for m in llm_info["models"]
        ]
        return models

    def get_default_model(self) -> str:
        """return the default model"""
        return DEFAULT_MODEL

    def load_model(self, model=None):
        """Load the model"""
        self.model_name = model or self.get_default_model()
        llm_key, model_name = self.model_name.split(":")
        llm_info = self.model_map.get(llm_key)
        self.llm = llm_info["class"](model_name=model_name)
        # asyncio.run(self.llm.load())

    def get_model_params(self, model_name):
        return self.llm.get_params()

    async def run_stream(
        self,
        prompt_value: PromptValue,
        verbose=True,
        **kwargs,
    ) -> AsyncGenerator[Any | str, Any]:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec
        if verbose:
            print("=" * 130)
            print(f"Prompt = {type(prompt_value)} / {prompt_value}")
            print(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            stream = await self.llm.generate_stream(prompt_value, verbose=verbose, **kwargs)
            async for response_delta in stream:
                response_text = response_delta.get_first_sequence()
                if response_text:
                    yield "content", response_text
                elif not response_text and response_delta.finish_reason:
                    yield "warning", "No response from LLM"

            if verbose:
                response_delta.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            # TODO: Can't do gradio specific in this class!
            yield "error", f"Unable to generate response [{e}]"

    async def run_batch(
        self,
        prompt_value: PromptValue,
        verbose=True,
        **kwargs,
    ) -> Any:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec
        if verbose:
            print("=" * 130)
            print(f"Prompt = {prompt_value}")
            print(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            llm_response = await self.llm.generate(prompt_value, verbose=verbose, **kwargs)
            response_text = llm_response.get_first_sequence()
            if verbose:
                llm_response.print_summary()

            if not response_text:
                return "warning", "No response from LLM"

            return "content", response_text
        except Exception as e:
            logger.warning(f"Exception = {e}")
            return "error", f"Unable to generate response [{e}]"
