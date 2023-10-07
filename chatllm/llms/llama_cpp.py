"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import glob
import os
from typing import Any, AsyncGenerator, Generator, List, Optional, Tuple

from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompt import PromptValue
from llama_cpp import Llama


@LLMRegister("llama-cpp")
class LlamaCpp(BaseLLMProvider):
    """Class for interfacing with Llama.cpp GGUF models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        model_dir = os.environ["CHATLLM_ROOT"] + "/models"
        model_path = f"{model_dir}/{model_name}"
        self.llm = Llama(model_path=model_path, n_ctx=2048)

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_params(self) -> List[str]:
        """Return Parameters supported by the model"""
        return {
            "max_tokens": 2500,
            "temperature": 0.8,
            "top_k": 3,
            "top_p": 0.9,
        }

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.llm.tokenize(bytes(prompt, encoding="utf-8"))
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple(str, int):
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
    ) -> List[str]:
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        result = self.llm(
            formatted_prompt,
            # stop=["Question:"],
            stream=False,  # Dont stream the response
            echo=False,  # Dont echo the prompt
            **kwargs,
        )
        if result.get("choices", []):
            first_resp = result["choices"][0]
            response_text = first_resp.pop("text")
            first_resp["message"] = {"role": "ai", "content": response_text}
            response_text = response_text.replace(formatted_prompt, "")
        return result

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        stream = self.llm(
            formatted_prompt,
            # stop=["Question:"],
            stream=True,  # Stream the response
            echo=False,  # Dont echo the prompt
            **kwargs,
        )

        # Wrap it in an async_generator!
        async def async_generator() -> Generator[Any]:
            llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
            num_sequences = -1
            for chunk in stream:
                generated_options = [c["text"] for c in chunk["choices"]]
                if num_sequences < 0:
                    num_sequences = len(generated_options)
                result = llm_response.add_delta(generated_options)
                yield result

            # Last token!
            result = llm_response.get_last_delta(num_deltas=num_sequences)
            print(f"Final result = {result}")
            yield result

        return async_generator()
