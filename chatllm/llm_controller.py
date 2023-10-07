"""Interface from Gradio/CLI to the LLM models."""
import json
import logging
import time
from typing import Any, Generator, List

from chatllm.chat_message import ChatMessage
from chatllm.llms.base import BaseLLMProvider
from chatllm.prompt import ChatPromptValue, PromptValue, StringPromptValue

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai:gpt-3.5-turbo"


class LLMController:
    """A chat Controller for conversational models."""

    def __init__(self) -> None:
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
        model = model or self.get_default_model()
        llm_key, model_name = model.split(":")
        llm_info = self.model_map.get(llm_key)
        self.llm = llm_info["class"](model_name=model_name)
        # asyncio.run(self.llm.load())

    def get_model_params(self, model_name):
        return self.llm.get_params()

    def create_prompt_value(self, user_query, system_prompt, chat_history) -> PromptValue:
        """Create a PromptValue object"""
        if system_prompt or len(chat_history) > 1:
            prompt_value = ChatPromptValue()
            if system_prompt:
                prompt_value.add_message(ChatMessage(role="system", content=system_prompt))
            for user_msg, ai_msg in chat_history:
                if user_msg:
                    prompt_value.add_message(ChatMessage(role="user", content=user_msg))
                if ai_msg:
                    prompt_value.add_message(ChatMessage(role="assistant", content=ai_msg))
            # User Query is included in the chat history.. No need to add it...
            # prompt_value.add_message(ChatMessage(role="user", content=user_query))
        else:
            prompt_value = StringPromptValue(text=user_query)
        return prompt_value

    async def run_stream(
        self,
        question,
        system_prompt="",
        chat_history=[],
        verbose=True,
        **kwargs,
    ) -> Generator[Any | str, Any, None]:
        start_time = time.time()
        prompt_value = self.create_prompt_value(question, system_prompt, chat_history)
        if verbose:
            print("=" * 130)
            print(f"Prompt = {type(prompt_value)} / {prompt_value}")
            print(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            stream = await self.llm.generate_stream(prompt_value, verbose=verbose, **kwargs)
            response_text = ""
            first_token_time = None
            async for response_delta in stream:
                if response_delta.get("choices", []):
                    first_resp = response_delta["choices"][0]
                    text = first_resp.get("delta", {}).get("content", "")

                    response_text += text
                    if first_token_time is None:
                        first_token_time = time.time()
                    print(f"Delta = {response_delta}")
                    yield response_text
            elapsed_time = time.time() - start_time
            token_gen_time = time.time() - first_token_time

            if verbose:
                usage = response_delta.get("usage", {})
                print(f"Response: {response_text}")
                print(f"    Model: {response_delta.get('model', 'n/a')}")
                print(f"    Usage = {json.dumps(usage)}")
                print(f"    Stop Reason = {first_resp.get('finish_reason', 'n/a')}")
                tokens_per_sec = (
                    f"{(usage['completion_tokens'] / elapsed_time):.02f} Tokens/Sec"
                    if usage
                    else "Not Available"
                )
                token_gen_str = (
                    f"Time between first token and last token: {token_gen_time:.03f} secs"
                )
                print(
                    f"Elapsed time = {elapsed_time:.03f} secs, {tokens_per_sec}. {token_gen_str}"
                )
                print("=" * 130)
        except Exception as e:
            logger.warn(f"Exception = {e}")
            yield f"<span style='color:red'>*Error: Unable to generate response* [{e}]</span>"

    async def run_batch(
        self,
        question,
        system_prompt="",
        chat_history=[],
        verbose=True,
        **kwargs,
    ) -> Any:
        start_time = time.time()
        prompt_value = self.create_prompt_value(question, system_prompt, chat_history)
        if verbose:
            print("=" * 130)
            print(f"Prompt = {prompt_value}")
            print(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            result = await self.llm.generate(prompt_value, verbose=verbose, **kwargs)
            elapsed_time = time.time() - start_time

            if result.get("choices", []):
                first_resp = result["choices"][0]
                response_text = first_resp.get("message", {}).get("content", "")
            else:
                response_text = (
                    f"<span style='color:red'>*Warning: No response from LLM* [{e}]</span>"
                )

            if verbose:
                usage = result.get("usage", {})
                print(f"Response: {response_text}")
                print(f"    Model: {result.get('model', 'n/a')}")
                print(f"    Usage = {json.dumps(usage or {})}")
                print(f"    Stop Reason = {first_resp.get('finish_reason', 'n/a')}")
                if "metrics" in result:
                    print(f"    Metrics = {json.dumps(result['metrics'])}")
                tokens_per_sec = (
                    f"{(usage['completion_tokens'] / elapsed_time):.02f} Tokens/Sec"
                    if usage
                    else "Not Available"
                )
                print(f"Elapsed time = {elapsed_time:.03f} secs, {tokens_per_sec}")
                print("=" * 130)
        except Exception as e:
            logger.warn(f"Exception = {e}")
            response_text = (
                f"<span style='color:red'>*Error: Unable to generate response* [{e}]</span>"
            )
        return response_text
