"""Language Model To Interface With Hugging Face Models"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, List, Optional, Tuple

import torch
import transformers
from chatllm.llms.base import BaseLLMProvider, LLMRegister


@LLMRegister("hf")
class HFPipeline(BaseLLMProvider):
    """Class for interfacing with HF models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        # self.pipeline= pipeline(model="gpt2")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            # model_kwargs={"load_in_8bit": True},
            torch_dtype=torch.float32 if model_name == "gpt2" else torch.float16,
            # device_map="cpu",
        )
        self.enc = self.pipeline.tokenizer

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
            "length_penalty": 1,
        }

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = self.enc(prompt)["input_ids"]
        return len(tokens)

    def format_prompt(self, prompt: str) -> Tuple(str, int):
        """Format the prompt and return the number of tokens in the prompt."""
        formatted_prompt = f"Question: {prompt} Answer: " if prompt else ""
        num_tokens = self.get_token_count(formatted_prompt)
        return formatted_prompt, num_tokens

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return a list of supported models."""
        return ["gpt2"]  # , "TheBloke/Llama-2-7B-fp16"]

    async def generate(
        self,
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        formatted_prompt, num_tokens = self.format_prompt(input_prompt)
        hf_response = self.pipeline(formatted_prompt)
        result = self.get_response_template(num_prompt_tokens=num_tokens, usage=True)
        if hf_response:
            ## Strip Question and NBSPs(0xa0)
            response_texts = (
                hf_response[0]["generated_text"].replace(formatted_prompt, "").replace("\xa0", "")
            )
            out_tokens = self.get_token_count(hf_response[0]["generated_text"])
            result["usage"]["completion_tokens"] = out_tokens
            result["usage"]["total_tokens"] = num_tokens + out_tokens
            result["choices"] = self.format_choice(response_texts)

        return result

    async def generate_stream(
        self,
        input_prompt: str,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """
        Pass a single prompt value to the model and stream model generations.
        Streaming not supported for HF models, so we similute a token-by-token async generation
        from the entire result
        """
        print("Streaming not supported for HF models, Using generate instead")
        formatted_prompt, num_tokens = self.format_prompt(input_prompt)

        async def async_generator() -> Generator[Any]:
            hf_response = self.pipeline(formatted_prompt)
            result = self.get_response_template(num_prompt_tokens=num_tokens, usage=False)
            if hf_response:
                ## Strip Question and NBSPs(0xa0)
                response_text = (
                    hf_response[0]["generated_text"]
                    .replace(formatted_prompt, "")
                    .replace("\xa0", "")
                )
                out_tokens = self.get_token_count(response_text)
                print(f"Number of output tokens from {self.model_name} = {out_tokens}")
                tokens = self.enc.batch_decode(self.enc(response_text)["input_ids"])
                token = tokens.pop(0) if tokens else None
                while token:
                    result["choices"] = self.format_delta(token)
                    token = tokens.pop(0) if tokens else None
                    yield result

            # Last token!
            result["choices"] = self.format_last_delta()
            yield result

        return async_generator()
