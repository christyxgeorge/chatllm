import json
import logging

from typing import Any, AsyncGenerator, Dict

from pydantic import BaseModel

from chatllm.llm_params import LLMConfig, LLMParam
from chatllm.prompts import (
    ChatMessage,
    ChatPromptValue,
    ChatRole,
    PromptValue,
    StringPromptValue,
)
from chatllm.prompts.default_prompts import simple_system_prompt

logger = logging.getLogger(__name__)


class LLMHistoryItem(BaseModel):
    text: str
    role: ChatRole


class LLMSession:
    chat_history: list[LLMHistoryItem] = []

    def __init__(self, llm, model_name, model_cfg) -> None:
        self.llm = llm
        self.model_name = model_name
        self.model_config: LLMConfig | None = model_cfg
        self.system_prompt_type = "simple"
        self.system_prompt = simple_system_prompt

    def get_model_params(self) -> Dict[str, LLMParam]:
        assert self.model_config is not None, f"Model {self.model_name} not loaded"  # nosec
        return self.model_config.get_params()

    def add_history(self, text: str, role: ChatRole) -> None:
        self.chat_history.append(LLMHistoryItem(text=text, role=role))

    def clear_history(self) -> None:
        self.chat_history = []

    def set_system_prompt(self, type: str, prompt: str = "") -> None:
        """Set the system prompt"""
        if prompt:
            self.system_prompt_type = type
            self.system_prompt = prompt
        else:
            self.system_prompt_type = "none"
            self.system_prompt = ""

    def create_prompt_value(self, user_query) -> PromptValue:
        """Create a PromptValue object"""
        if self.system_prompt:
            prompt_value = ChatPromptValue()
            if self.system_prompt or self.chat_history:
                prompt_value.add_message(
                    ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)
                )
            for msg in self.chat_history:
                prompt_value.add_message(ChatMessage(role=msg.role, content=msg.text))
            # Add the User Query
            prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_query))
            return prompt_value
        else:
            sprompt_value = StringPromptValue(text=user_query)
            return sprompt_value

    async def run_stream(
        self,
        user_query: str,
        verbose=False,
        word_by_word=False,
        **kwargs,
    ) -> AsyncGenerator[Any | str, Any]:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec

        # TODO: Need to move this out to the CLI/Gradio layer?
        if verbose:
            logger.info("=" * 50)
            logger.info(f"Prompt = {user_query}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            prompt_value: PromptValue = self.create_prompt_value(user_query)
            self.add_history(user_query, role=ChatRole.USER)
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

            yield "done", ""
            self.add_history(llm_response.get_first_sequence(), role=ChatRole.AI)
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            # TODO: Can't do gradio specific in this class!
            yield "error", f"Unable to generate response [{e}]"

    async def run_batch(
        self,
        user_query: str,
        verbose=False,
        **kwargs,
    ) -> Any:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec
        if verbose:
            logger.info("=" * 50)
            logger.info(f"User Query = {user_query}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            prompt_value: PromptValue = self.create_prompt_value(user_query)
            self.add_history(user_query, role=ChatRole.USER)
            llm_response = await self.llm.generate(prompt_value, verbose=verbose, **kwargs)
            response_text = llm_response.get_first_sequence()
            if not response_text:
                yield "warning", "No response from LLM"
            else:
                yield "content", response_text

            yield "done", ""
            self.add_history(llm_response.get_first_sequence(), role=ChatRole.AI)
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            yield "error", f"Unable to generate response [{e}]"
