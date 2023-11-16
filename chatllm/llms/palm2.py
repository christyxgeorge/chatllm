"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import logging
import os

from typing import Any, AsyncGenerator, List, Tuple

import google.generativeai as palm

from pydantic import model_validator

from chatllm.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMParam,
    MaxTokens,
    NumSequences,
    RepeatPenalty,
    Temperature,
    TopK,
)
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
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

    @model_validator(mode="after")
    @classmethod
    def setup_palm2_model_info(cls, pconfig: Any) -> Any:
        """Validate the config"""
        palm_models = Palm2Api.get_palm_models()
        pmodel_prefix = pconfig.name.replace("palm2:", "models/")
        palm_model = [pm for pm in palm_models if pm.name.startswith(pmodel_prefix)][0]
        if palm_model:
            pconfig.palm_model_name = palm_model.name
            pconfig.palm_supported_methods = palm_model.supported_generation_methods
        return pconfig


@LLMRegister(config_class=Palm2Config)
class Palm2Api(BaseLLMProvider):
    """Class for interfacing with GCP PaLM2 models."""

    palm_models: List[Any] = []

    def __init__(self, model_name: str, model_cfg: LLMConfig, **kwargs) -> None:
        super().__init__(model_name, model_cfg, **kwargs)
        # credentials_file = os.environ.get("GCLOUD_CREDENTIALS_FILE")
        # credentials = service_account.Credentials.from_service_account_file(credentials_file)
        # palm.configure(credentials=credentials)
        palm.configure(api_key=os.environ["PALM2_API_KEY"])
        self.llm = palm
        assert isinstance(model_cfg, Palm2Config), "Configuration File not PALM Specific"  # nosec
        self.palm_model_name = model_cfg.palm_model_name
        self.supported_methods = model_cfg.palm_supported_methods
        logger.info(f"Supported Methods: {self.supported_methods}")

    @staticmethod
    def get_palm_models() -> List[Any]:
        """Return a list of supported models."""
        if not Palm2Api.palm_models:
            palm.configure(api_key=os.environ["PALM2_API_KEY"])
            pmodels = palm.list_models()
            # Convert the Generator Object to a list
            Palm2Api.palm_models = [m for m in pmodels]
        return Palm2Api.palm_models

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[str]:
        """Return a list of supported models."""

        palm_models = Palm2Api.get_palm_models()
        if verbose:
            logger.info(f"PALM2 Model List = {palm_models}")
        models = [m.name.replace("models/", "") for m in palm_models]
        return models

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of Palm2
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
