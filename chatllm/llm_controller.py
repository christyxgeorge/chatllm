"""Interface from Gradio/CLI to the LLM models."""
import json
import logging
import time
from typing import Any, List

from chatllm.llms.base import BaseLLMProvider

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

    async def run_stream(
        self,
        question,
        system_prompt="",
        verbose=True,
        **kwargs,
    ):
        if verbose:
            print("=" * 130)
            print(f"Question: {question}")
            print(f"    Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        start_time = time.time()
        try:
            prompt_tokens = self.llm.get_token_count(question)
            stream = await self.llm.generate_stream(question, verbose=verbose, **kwargs)
            response_text = ""
            completion_tokens = 0
            async for chunk in stream:
                if chunk.get("choices", []):
                    first_resp = chunk["choices"][0]
                    text = first_resp.get("delta", {}).get("content", "")

                    response_text += text
                    # if completion_tokens == 0:
                    #     first_token_time = time.time()
                    completion_tokens += 1
                    yield response_text
            elapsed_time = time.time() - start_time
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            if verbose:
                print(f"Response: {response_text}")
                print(f"    Model: {chunk.get('model', 'n/a')}")
                print(f"    Usage = {json.dumps(usage or {})}")
                print(f"    Stop Reason = {first_resp.get('finish_reason', 'n/a')}")
                tokens_per_sec = (
                    f"{(usage['completion_tokens'] / elapsed_time):.02f}"
                    if usage
                    else "Not Available"
                )
                print(f"Elapsed time = {elapsed_time:.03f} secs, {tokens_per_sec} Tokens/Sec")
                print("=" * 130)
        except Exception as e:
            print(f"Exception = {e}")
            yield f"<span style='color:red'>*Error: Unable to generate response* [{e}]</span>"

    async def run_query(
        self,
        question,
        system_prompt="",
        verbose=True,
        **kwargs,
    ) -> Any:
        if verbose:
            print("=" * 130)
            print(f"Question: {question}")
            print(f"    Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        start_time = time.time()
        prompt_tokens = self.llm.get_token_count(question)
        result = await self.llm.generate(question, verbose=verbose, **kwargs)
        elapsed_time = time.time() - start_time

        if result.get("choices", []):
            first_resp = result["choices"][0]
            usage = result.get("usage")
            usage["tiktokens"] = prompt_tokens
            response_text = first_resp.get("message", {}).get("content", "")
        else:
            response_text = "Unable to generate response"

        if verbose:
            print(f"Response: {response_text}")
            print(f"    Model: {result.get('model', 'n/a')}")
            print(f"    Usage = {json.dumps(usage or {})}")
            print(f"    Stop Reason = {first_resp.get('finish_reason', 'n/a')}")
            if "metrics" in result:
                print(f"    Metrics = {json.dumps(result['metrics'])}")
            tokens_per_sec = (
                f"{(usage['completion_tokens'] / elapsed_time):.02f}" if usage else "Not Available"
            )
            print(f"Elapsed time = {elapsed_time:.03f} secs, {tokens_per_sec} Tokens/Sec")
            print("=" * 130)
        # TODO: Add verbose levels.
        # debug = False
        # if debug:
        # print(f"Result = {result}")
        return response_text
