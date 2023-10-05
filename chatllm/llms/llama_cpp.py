"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import glob
import os
from typing import Any, AsyncGenerator, Generator, List, Optional

from chatllm.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from chatllm.llms.base import BaseLLMProvider, LLMRegister
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

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        formatted_prompt = self.format_prompt(prompt)
        tokens = self.llm.tokenize(bytes(formatted_prompt, encoding="utf-8"))
        # print(f"Encoding = {prompt} // {tokens}")
        return len(tokens)

    def format_prompt(self, prompt: str) -> str:
        """Format the prompt for Llama CPP"""
        return f"Question: {prompt} Answer: " if prompt else ""

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
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        formatted_prompt = self.format_prompt(input_prompt)
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
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        qtemplate = f"Question: {input_prompt} Answer: "
        num_tokens = self.get_token_count(qtemplate)
        stream = self.llm(
            qtemplate,
            # stop=["Question:"],
            stream=True,  # Stream the response
            echo=False,  # Dont echo the prompt
            **kwargs,
        )

        # Wrap it in an async_generator!
        async def async_generator() -> Generator[Any]:
            result = self.get_response_template(num_prompt_tokens=num_tokens, usage=False)
            for chunk in stream:
                generated_options = [c["text"] for c in chunk["choices"]]
                result["choices"] = self.format_delta(generated_options)
                yield result

            # Last token!
            result["choices"] = self.format_last_delta()
            yield result

        return async_generator()
