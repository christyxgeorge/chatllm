"""Interface from Gradio/CLI/PyTest to use the LLM models."""
import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional

from chatllm.llms.base import BaseLLMProvider
from chatllm.prompts import PromptValue
from chatllm.prompts.chat_message import ChatMessage, ChatRole
from chatllm.prompts.prompt_value import ChatPromptValue, StringPromptValue

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

    def get_provider_model_list(self, provider):
        """return the list of models for the specified provider"""
        models = [
            f"{llm_key}:{m}"
            for llm_key, llm_info in self.model_map.items()
            for m in llm_info["models"]
            if llm_key == provider
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

    def create_prompt_value(self, user_query, system_prompt, chat_history=[]) -> PromptValue:
        """Create a PromptValue object"""
        prompt_value: Optional[PromptValue] = None
        if system_prompt or len(chat_history) > 1:
            prompt_value = ChatPromptValue()
            if system_prompt:
                prompt_value.add_message(ChatMessage(role=ChatRole.SYSTEM, content=system_prompt))
            for user_msg, ai_msg in chat_history:
                if user_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_msg))
                if ai_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.AI, content=ai_msg))
            if not chat_history:
                # User Query is included in the chat history.. Add only when there is no chat_history
                prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_query))
        else:
            prompt_value = StringPromptValue(text=user_query)
        return prompt_value

    async def run_stream(
        self,
        prompt_value: PromptValue,
        verbose=False,
        word_by_word=False,
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
                response_text = (
                    response_delta.get_first_of_last_token()
                    if word_by_word
                    else response_delta.get_first_sequence()
                )
                if response_text:
                    yield "content", response_text
                elif response_delta.finish_reason:
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
        verbose=False,
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
