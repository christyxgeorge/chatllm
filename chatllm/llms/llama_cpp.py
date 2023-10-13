"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import glob
import os
from typing import Any, AsyncGenerator, Dict, List, Tuple, cast

from llama_cpp import Completion, CompletionChoice, CompletionChunk, Llama

from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompts import PromptValue


@LLMRegister("llama-cpp")
class LlamaCpp(BaseLLMProvider):
    """Class for interfacing with Llama.cpp GGUF models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        model_dir = os.environ["CHATLLM_ROOT"] + "/models"
        model_path = f"{model_dir}/{model_name}"
        # Note: Default context in llama-cpp is 512!
        self.llm = Llama(model_path=model_path, n_ctx=2048)
        self.num_sequences = 1

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_params(self) -> Dict[str, float | object]:
        """
        Return Parameters supported by the model
        Since we are generating locally, by default, we dont need to limit the tokens
        """
        return {
            "max_tokens": 0,
            "temperature": 0.8,
            "top_k": 3,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.llm.tokenize(bytes(prompt, encoding="utf-8"))
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple[str, int]:
        """Format the prompt and return the number of tokens in the prompt."""
        # formatted_prompt = f"Question: {prompt} Answer: " if prompt else ""
        formatted_prompt = prompt_value.to_string()
        return formatted_prompt, self.get_token_count(formatted_prompt)

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return a list of supported models."""
        model_dir = os.environ["CHATLLM_ROOT"] + "/models"
        data_glob = os.path.join(model_dir, "*.gguf")
        files = sorted(glob.glob(data_glob))
        # print(f"glob = {data_glob}, Files = {len(files)}")
        return [os.path.basename(f) for f in files]

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:  # Generator[LLMResponse, Any, Any]:
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        llm_response = LLMResponse(
            model=self.model, num_sequences=self.num_sequences, prompt_tokens=num_tokens
        )
        response = self.llm(
            formatted_prompt,
            # stop=["Question:"],
            stream=False,  # Dont stream the response
            echo=False,  # Dont echo the prompt
            **kwargs,
        )
        # Reformat choices to openai format!
        response = cast(Completion, response)
        for r in response["choices"]:
            r["message"] = {"role": "assistant", "content": r.pop("text")}  # type: ignore
        llm_response.set_openai_response(response)  # type: ignore
        return llm_response

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any | str, Any]:
        """Pass a single prompt value to the model and stream model generations."""
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        llm_response = LLMResponse(
            model=self.model, num_sequences=self.num_sequences, prompt_tokens=num_tokens
        )
        stream = self.llm(
            formatted_prompt,
            # stop=["Question:"],
            stream=True,  # Stream the response
            echo=False,  # Dont echo the prompt
            **kwargs,
        )

        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            for response_delta in stream:
                response_delta = cast(CompletionChunk, response_delta)
                # Reformat choices to openai format!
                for r in response_delta["choices"]:
                    r = cast(CompletionChoice, r)
                    r["delta"] = {"content": r.pop("text")}  # type: ignore
                llm_response.add_openai_delta(cast(Dict[str, Any], response_delta))
                yield llm_response

            # Last token [Llama-cpp does not seem to handle 'length']
            if llm_response.completion_tokens >= kwargs.get("max_tokens", 0):
                finish_reason = "length"
            else:
                finish_reason = "stop"

            llm_response.add_last_delta(finish_reason=finish_reason)
            yield llm_response

        return async_generator()
