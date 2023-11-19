import json
import logging

from typing import Any, AsyncGenerator, Dict, Optional

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
    role: str
    token_count: int


class LLMSession:
    entries: list[LLMHistoryItem] = []

    def __init__(self, llm, model_name, model_cfg) -> None:
        self.llm = llm
        self.model_name = model_name
        self.model_config: LLMConfig | None = model_cfg
        self.system_prompt_type = "simple"
        self.system_prompt = simple_system_prompt

    def get_model_params(self) -> Dict[str, LLMParam]:
        assert self.model_config is not None, f"Model {self.model_name} not loaded"  # nosec
        return self.model_config.get_params()

    def add_history(self, text: str, role: str, token_count: int) -> None:
        self.entries.append(LLMHistoryItem(text=text, role=role, token_count=token_count))

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

            yield "done", ""
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
            if not response_text:
                yield "warning", "No response from LLM"
            else:
                yield "content", response_text
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            yield "error", f"Unable to generate response [{e}]"
