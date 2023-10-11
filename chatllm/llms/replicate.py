"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, List, Tuple

import replicate
import requests
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


@LLMRegister("replicate")
class ReplicateApi(BaseLLMProvider):
    """Class for interfacing with Llama.cpp GGUF models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

        rep_model = replicate.models.get(model_name)
        model_version = rep_model.versions.list()[0]  ### Get the latest version
        properties = model_version.openapi_schema["components"]["schemas"]["Input"]["properties"]
        self.llm = model_version
        self.input_properties = {k: v for k, v in properties.items()}
        # print(f"Properties = {self.input_properties}")

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_params(self) -> List[str]:
        """Return Parameters supported by the model"""
        return {
            "max_tokens": 2500,  # max_length
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1,  # repetition_penalty
        }

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return a list of supported models."""
        llm_models_url = "https://api.replicate.com/v1/collections/language-models"
        api_key = os.environ.get("REPLICATE_API_TOKEN")
        response = requests.get(
            llm_models_url, timeout=30, headers={"Authorization": f"Token {api_key}"}
        )
        if response.status_code == 200:
            models = [f"{x['owner']}/{x['name']}" for x in response.json()["models"]]
        else:
            print("Unable to get LLM models from Replicate")
            models = []
        return models

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens = []
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple(str, int):
        """Format the prompt for Replicate Predictions: Nothing to be done!"""
        formatted_prompt = prompt_value.to_string()
        return formatted_prompt, self.get_token_count(formatted_prompt)

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs passed to the model"""
        validated_kwargs = {}
        # Rename max_tokens to max_length
        kwargs["max_length"] = kwargs.pop("max_tokens", 2500)  # Rename to max_new_tokens
        kwargs["repetition_penalty"] = kwargs.pop("repeat_penalty", 1)
        logger.info(f"Supported tokens: {', '.join(k for k in self.input_properties.keys())}")
        for k, v in kwargs.items():
            if k not in self.input_properties:
                logger.warning(f"Invalid key {k} for model {self.model_name}, Ignoring")
            else:
                validated_kwargs[k] = v
        return validated_kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
        kwargs = self.validate_kwargs(**kwargs)
        input_args = {
            "prompt": formatted_prompt,
            # "stop_sequences": stop,
            "debug": verbose,
            "stream": True,
            "xxxx": 1,  ## Just to check the OpenAPI validaton
            **kwargs,
        }

        prediction = replicate.predictions.create(version=self.llm, input=input_args)
        iterator = prediction.output_iterator()

        out_tokens = 0
        response_text = ""
        for chunk in iterator:
            out_tokens += 1
            response_text += chunk

        llm_response.set_response(response_text, "stop")
        llm_response.set_token_count(num_tokens, out_tokens)
        self.set_prediction_info(prediction, llm_response)
        return llm_response

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
        kwargs = self.validate_kwargs(**kwargs)
        input_args = {
            "prompt": formatted_prompt,
            # "stop_sequences": stop,
            "debug": verbose,
            "stream": True,
            "xxxx": 1,  ## Just to check the validaton
            **kwargs,
        }

        prediction = replicate.predictions.create(version=self.llm, input=input_args)

        # TODO: Validation of the args is not being done!
        print(f"Replicate prediction using {self.model_name}; Args = {input_args}")

        # Wrap it in an async_generator!
        async def async_generator():
            iterator = prediction.output_iterator()
            for text_chunk in iterator:
                llm_response.add_delta(text_chunk)
                yield llm_response

            # Last token!
            self.set_prediction_info(prediction, llm_response)
            llm_response.add_last_delta()
            yield llm_response

        return async_generator()

    def set_prediction_info(self, prediction, llm_response: LLMResponse):
        """Sets the info for the generation in LLMResponse"""
        try:
            prediction_logs = {
                k[0]: k[1].strip()
                for k in (x.split(":") for x in prediction.logs.split("\n") if ":" in x)
            }
            logger.info(f"Prediction Status = {prediction.status}")
            logger.info(f"Prediction Logs = {prediction_logs}")
            prompt_tokens = int(prediction_logs.get("Number of tokens in prompt", 0))
            completion_tokens = int(prediction_logs.get("Number of tokens generated", 0))
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            llm_response.set_api_usage(usage)
            llm_response.set_extra_info(metrics=prediction.metrics)

        except Exception as e:
            logger.warning(f"Exception while parsing prediction logs: {e}")
