"""Interface from Gradio/CLI/PyTest to use the LLM models."""
import functools
import json
import logging

from typing import Any, AsyncGenerator, Dict, List, Optional

from chatllm.llm_params import LLMConfig, LLMParam
from chatllm.llms import PROVIDER_ORDER
from chatllm.llms.base import BaseLLMProvider
from chatllm.prompts import (
    ChatMessage,
    ChatPromptValue,
    ChatRole,
    PromptValue,
    StringPromptValue,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "open_ai:gpt-3.5-turbo"

simple_system_prompt = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe.\
"""
long_system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,
while being safe. Your answers should not include any harmful, unethical, racist, sexist,
toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased
and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of
answering something not correct. If you don't know the answer to a question, please don't share
false information.
"""


class LLMController:
    """A chat Controller for conversational models."""

    def __init__(self) -> None:
        self.model_name = None
        self.llm = None
        self.model_config: LLMConfig | None = None
        self.system_prompt_type = "simple"
        self.system_prompt = simple_system_prompt

    def sortby_provider(self, x, y) -> int:
        prov_x = x.split(":")[0]
        prov_y = y.split(":")[0]
        if prov_x == prov_y:
            return 1 if x > y else (0 if x == y else -1)
        else:
            idx_x = (
                PROVIDER_ORDER.index(prov_x) if prov_x in PROVIDER_ORDER else len(PROVIDER_ORDER)
            )
            idx_y = (
                PROVIDER_ORDER.index(prov_y) if prov_y in PROVIDER_ORDER else len(PROVIDER_ORDER)
            )
            return idx_x - idx_y

    def get_model_list(self) -> List[str]:
        """return the list of models"""
        model_map = BaseLLMProvider.registered_models()
        models = [
            f"{llm_key}:{m.name}"
            for llm_key, llm_info in model_map.items()
            for m in llm_info["models"]
        ]
        sorted_models = sorted(models, key=functools.cmp_to_key(self.sortby_provider))
        return sorted_models

    def get_provider_model_list(self, provider) -> List[str]:
        """return the list of models for the specified provider"""
        model_map = BaseLLMProvider.registered_models()

        models = [
            f"{llm_key}:{m}"
            for llm_key, llm_info in model_map.items()
            for m in llm_info["models"]
            if llm_key == provider
        ]
        return models

    @staticmethod
    def get_model_key_map() -> Dict[str, str]:
        """Return model key to model name mapping (for use in CLI)"""
        model_map = BaseLLMProvider.registered_models()
        model_key_map = {
            m.key: f"{llm_key}:{m.name}"
            for llm_key, llm_info in model_map.items()
            for m in llm_info["models"]
            if m.key
        }
        return model_key_map

    def get_default_model(self) -> str:
        """return the default model"""
        return DEFAULT_MODEL

    def load_model(self, model=None):
        """Load the model"""
        self.model_name = model or self.get_default_model()
        llm_key, model_name = self.model_name.split(":")
        model_map = BaseLLMProvider.registered_models()
        llm_info = model_map.get(llm_key)
        self.llm = llm_info["class"](model_name=model_name)
        self.model_config = [mcfg for mcfg in llm_info["models"] if mcfg.name == model_name][0]
        logger.info(f"Loaded Model: {llm_key}:{model_name}")
        # asyncio.run(self.llm.load())

    def get_model_params(self) -> Dict[str, LLMParam]:
        assert self.model_config is not None, f"Model {self.model_name} not loaded"  # nosec
        return self.model_config.get_params()

    def get_system_prompt_list(self) -> Dict[str, str]:
        """return the list of system prompts"""
        return {
            "simple": simple_system_prompt,
            "long": long_system_prompt,
            "none": "",
            "custom": "",
        }

    def set_system_prompt(self, type: str, prompt: str = "") -> None:
        """Set the system prompt"""
        if prompt:
            self.system_prompt_type = type
            self.system_prompt = prompt
        else:
            self.system_prompt_type = "none"
            self.system_prompt = ""

    def create_prompt_value(self, user_query, chat_history=[]) -> PromptValue:
        """Create a PromptValue object"""
        prompt_value: Optional[PromptValue] = None
        if self.system_prompt or len(chat_history) > 1:
            prompt_value = ChatPromptValue()
            if self.system_prompt:
                prompt_value.add_message(
                    ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)
                )
            for user_msg, ai_msg in chat_history:
                if user_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_msg))
                if ai_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.AI, content=ai_msg))
            if not chat_history:
                # User Query is included in the chat history.. Add only when there is no chat_history # noqa: E501
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
            logger.info("=" * 50)
            logger.info(f"Prompt = {type(prompt_value)} / {prompt_value}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            stream = await self.llm.generate_stream(prompt_value, verbose=verbose, **kwargs)
            async for llm_response in stream:
                response_text = (
                    llm_response.get_first_of_last_token()
                    if word_by_word
                    else llm_response.get_first_sequence()
                )
                if response_text:
                    yield "content", response_text
                elif llm_response.finish_reasons:
                    finish_reasons = "|".join(llm_response.finish_reasons)
                    yield "warning", f"No response from LLM [Reason = {finish_reasons}]"

            if verbose:
                llm_response.print_summary()

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
            logger.info("=" * 50)
            logger.info(f"Prompt = {prompt_value}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

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
