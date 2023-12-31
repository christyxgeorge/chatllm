"""Language Model To Interface With Hugging Face Models"""
from __future__ import annotations

import glob
import logging
import os

from typing import Any, AsyncGenerator, List

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from chatllm.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMParam,
    MaxTokens,
    RepeatPenalty,
    Temperature,
)
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


class HuggingFaceConfig(LLMConfig):
    # Reference: https://huggingface.co/docs/transformers/main_classes/text_generation
    max_tokens: LLMParam = MaxTokens(name="max_new_tokens")
    temperature: LLMParam = Temperature(min=0, max=2, default=1)
    length_penalty: LLMParam = LengthPenalty(min=0, max=2, default=1)
    repeat_penalty: LLMParam = RepeatPenalty(name="repetition_penalty", min=0, max=2, default=1)


@LLMRegister(config_class=HuggingFaceConfig)
class HFPipeline(BaseLLMProvider):
    """Class for interfacing with HF models."""

    def __init__(self, model_name: str, model_cfg: LLMConfig, **kwargs) -> None:
        super().__init__(model_name, model_cfg, **kwargs)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer initialized {self.tokenizer}")

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[str]:
        """Return a list of supported models."""
        model_dir = os.environ["HF_CACHE_DIR"]
        data_glob = os.path.join(model_dir, "models--*")
        files = sorted(glob.glob(data_glob))
        # print(f"glob = {data_glob}, Files = {len(files)}")
        models: List[str] = []
        for f in files:
            name = f"{os.path.basename(f)}".replace("models--", "").replace("--", "/")
            models.append(name)
        return models

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of Hugging face Hub
        as we load the model in the constructor.
        """
        pass

    def get_token_count(self, prompt: str) -> int:
        """
        Return the number of tokens in the prompt
        """
        tokens = self.tokenizer(prompt)["input_ids"]
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> str:
        """Format the prompt and return the number of tokens in the prompt."""
        # formatted_prompt = f"Question: {prompt} Answer: " if prompt else ""
        if self.model_name in ["roneneldan/TinyStories-33M"]:  # , 'replit/replit-code-v1_5-3b']:
            formatted_prompt = prompt_value.to_string(format="user_last")
        else:
            formatted_prompt = prompt_value.to_string(format="role:content")
        return formatted_prompt  # , self.get_token_count(formatted_prompt)

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs for the model"""
        kwargs["repetition_penalty"] = kwargs.pop("repeat_penalty", 1.0)
        if "max_tokens" in kwargs:
            # Rename to max_new_tokens
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")
        kwargs["do_sample"] = True
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # To check multiple sequences
        num_sequences = kwargs["num_return_sequences"] = kwargs.pop("num_sequences", 1)
        if num_sequences > 1:
            logger.info(f"Setting num_beams / num_return_sequences = {num_sequences}")
            kwargs["num_beams"] = num_sequences
        logger.info(f"Validated kwargs = {kwargs}")
        return kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        num_sequences = kwargs.get("num_return_sequences", 1)
        validated_kwargs = self.validate_kwargs(**kwargs)
        llm_response = LLMResponse(model=self.model, num_sequences=num_sequences)
        formatted_prompt = self.format_prompt(prompt_value)
        input_tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        num_tokens = torch.numel(input_tokens)

        hf_response = self.llm.generate(input_tokens, **validated_kwargs)
        out_tokens = torch.numel(hf_response)  # sum([len(rt) for rt in zipped_tokens])
        llm_response.set_token_count(prompt_count=num_tokens, completion_count=out_tokens)
        if out_tokens:
            response_texts = [
                "".join(self.tokenizer.batch_decode(seq, skip_special_tokens=True))
                for seq in hf_response
            ]
            llm_response.set_response(response_texts, finish_reasons=["stop"])
        return llm_response

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any | str, Any]:
        """
        Pass a single prompt value to the model and stream model generations.
        Streaming not supported for HF models, so we similute a token-by-token async generation
        from the entire result
        """
        logger.warning("Streaming not supported for HF models, Simulating generate instead")
        formatted_prompt = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        input_tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        num_tokens = torch.numel(input_tokens)
        num_sequences = kwargs.get("num_return_sequences", 1)
        llm_response = LLMResponse(
            model=self.model, num_sequences=num_sequences, prompt_tokens=num_tokens
        )

        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            hf_response = self.llm.generate(input_tokens, **validated_kwargs)
            out_tokens = torch.numel(hf_response)  # sum([len(rt) for rt in zipped_tokens])
            logger.info(f"Hugging Face Response = {out_tokens} tokens generated")
            if out_tokens:
                # Strip Question and NBSPs(0xa0)
                zipped_tokens = hf_response.t().tolist()
                for tokens in zipped_tokens:
                    token_texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                    llm_response.add_delta(list(token_texts))
                    yield llm_response

            # Last token [HF does not seem to handle 'length']
            if out_tokens >= kwargs.get("max_tokens", 0):
                finish_reason = "length"
            else:
                finish_reason = "stop"
            llm_response.add_last_delta(finish_reasons=[finish_reason])
            yield llm_response

        return async_generator()
