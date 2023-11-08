"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import logging
import os

from typing import Any, AsyncGenerator, List, Tuple, cast

import google.generativeai as palm

from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.llms.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMParam,
    MaxTokens,
    NumSequences,
    RepeatPenalty,
    Temperature,
    TopK,
)
from chatllm.prompts import PromptValue

# from google.oauth2 import service_account


logger = logging.getLogger(__name__)


class Palm2Config(LLMConfig):
    # https://developers.generativeai.google/api/python/google/generativeai
    # Handle parameter variations
    max_tokens: LLMParam = MaxTokens(
        name="max_output_tokens", min=0, max=1024, step=64, default=128
    )
    temperature: LLMParam = Temperature(min=0, max=1, default=0.75, step=0.05)
    length_penalty: LLMParam = LengthPenalty(active=False)
    repeat_penalty: LLMParam = RepeatPenalty(active=False)
    num_sequences: LLMParam = NumSequences(name="candidate_count", min=1, max=8, default=1)
    top_k: LLMParam = TopK(min=0, max=40, default=40, step=10)

    # Variables from the PALM ListModels API
    palm_model_name: str = ""
    palm_supported_methods: List[str] = []


PALM2_MODEL_LIST: List[Palm2Config] = [
    Palm2Config(name="chat-bison", key="pcb", desc="PaLM-2 for Chat", ctx=8192, cpt=0.0),
    Palm2Config(
        name="text-bison",
        key="ptb",
        desc="PaLM-2 for Text Generation",
        ctx=8192,
        cpt=0.0,
    ),
]


@LLMRegister()
class Palm2Api(BaseLLMProvider):
    """Class for interfacing with GCP PaLM2 models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        # credentials_file = os.environ.get("GCLOUD_CREDENTIALS_FILE")
        # credentials = service_account.Credentials.from_service_account_file(credentials_file)
        # palm.configure(credentials=credentials)
        palm.configure(api_key=os.environ["PALM2_API_KEY"])
        self.llm = palm
        palm_models = [m for m in PALM2_MODEL_LIST if m.name == self.model_name]
        palm_model = palm_models[0] if palm_models else None
        self.palm_model_name = palm_model.palm_model_name if palm_model else None
        self.supported_methods = palm_model.palm_supported_methods if palm_model else []
        logger.info(f"Supported Methods: {self.supported_methods}")

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[LLMConfig]:
        """Return a list of supported models."""
        palm.configure(api_key=os.environ["PALM2_API_KEY"])
        mlist = palm.list_models()
        model_list = [model for model in mlist]
        if verbose:
            logger.info(f"Model List = {model_list}")
        # TODO: Can use defaults for ctx, top_p, top_k, temp, etc. from the API response
        for model in PALM2_MODEL_LIST:
            palm_models = [m for m in model_list if m.name.startswith(f"models/{model.name}")]
            palm_model = palm_models[0] if palm_models else None
            if palm_model:
                model.palm_model_name = palm_model.name
                model.palm_supported_methods = palm_model.supported_generation_methods
        model_list = [m for m in PALM2_MODEL_LIST if m.palm_model_name]
        return cast(List[LLMConfig], model_list)

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        if prompt:
            count_method = (
                self.llm.count_text_tokens
                if ("countTextTokens" in self.supported_methods)
                else self.llm.count_message_tokens
            )
            token_count = count_method(model=self.palm_model_name, prompt=prompt)
            # print(f"Token Count for {prompt} = {len(prompt)} / {token_count}")
            return token_count.get("token_count", 0)
        else:
            return 0

    def format_prompt(self, prompt_value: PromptValue) -> Tuple[str, int]:
        """Format the prompt for Vertex AI Predictions: Nothing to be done!"""
        formatted_prompt = prompt_value.to_string()
        return formatted_prompt, self.get_token_count(formatted_prompt)

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs passed to the model"""
        # Rename max_tokens to max output tokens
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens", 128)
        kwargs["candidate_count"] = int(kwargs.pop("num_sequences", 1))
        return kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate Text
        Palm2 API has an async API only for chat!.
        """
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)

        if "generateText" in self.supported_methods:
            completion = self.llm.generate_text(
                model=self.palm_model_name, prompt=formatted_prompt, **validated_kwargs
            )
            result = completion.result
        else:
            validated_kwargs.pop("max_output_tokens")  # TODO: Check why we cannot send this?
            completion = await self.llm.chat_async(
                model=self.palm_model_name, prompt=formatted_prompt, **validated_kwargs
            )
            result = completion.last

        if result:
            completion_tokens = self.get_token_count(result)
            llm_response.set_response(result, ["stop"])
            llm_response.set_token_count(num_tokens, completion_tokens)
        else:
            # TODO: Need to handle 'BlockedReason.SAFETY' from completion.filters!
            logger.warning(f"No response from the model! {completion}")
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
        Note: PalM2 API does not have a streaming API, so we simulate streaming!
        """
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)

        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
        if "generateText" in self.supported_methods:
            completion = self.llm.generate_text(
                model=self.palm_model_name, prompt=formatted_prompt, **validated_kwargs
            )
            result = completion.result
        else:
            validated_kwargs.pop("max_output_tokens")  # TODO: Check why we cannot send this?
            completion = await self.llm.chat_async(
                model=self.palm_model_name, prompt=formatted_prompt, **validated_kwargs
            )
            result = completion.last

        # Wrap it in an async_generator!
        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            completion_tokens = self.get_token_count(result)
            if result:
                llm_response.add_delta(result)
                yield llm_response
            else:
                # TODO: Need to handle 'BlockedReason.SAFETY' from completion.filters!
                logger.warning(f"No response from the model! {completion}")

            # Last token!
            if llm_response.completion_tokens >= kwargs.get("max_tokens", 0):
                finish_reason = "length"
            else:
                finish_reason = "stop"
            llm_response.set_token_count(num_tokens, completion_tokens)
            llm_response.add_last_delta(finish_reasons=[finish_reason])
            yield llm_response

        return async_generator()
