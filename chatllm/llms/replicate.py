"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import os
from typing import Any, AsyncGenerator, List, Tuple
from urllib import response

import replicate
import requests
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompt import PromptValue


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
            "max_tokens": 2500,
            "temperature": 0.8,
            "top_k": 3,
            "top_p": 0.9,
            "length_penalty": 1,
        }

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return a list of supported models."""
        llm_models_url = "https://api.replicate.com/v1/collections/language-models"
        api_key = os.environ.get("REPLICATE_API_TOKEN")
        response = requests.get(llm_models_url, headers={"Authorization": f"Token {api_key}"})
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

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        input_args = {
            "prompt": formatted_prompt,
            # "stop_sequences": stop,
            "debug": verbose,
            "stream": True,
            "xxxx": 1,  ## Just to check the OpenAPI validaton
            **kwargs,
        }

        # TODO: Validation of the args is not being done!
        print(f"Replicate prediction using {self.model_name}; Args = {input_args}")
        iterator = self.llm.predict(**input_args)  # Iterator over the tokens

        out_tokens = 0
        response_text = ""
        for chunk in iterator:
            out_tokens += 1
            response_text += chunk

        llm_response = LLMResponse(
            model=self.model, prompt_tokens=num_tokens, completion_tokens=out_tokens
        )
        result = llm_response.get_result(response_text)
        return result

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs passed to the model"""
        validated_kwargs = {}
        for k, v in kwargs.items():
            print(f"Key = {k}, Value = {v}")
            if k not in self.input_properties:
                print(f"Invalid key {k} for model {self.model_name}, Ignoring")
            else:
                validated_kwargs[k] = v
        return validated_kwargs

    async def generate_stream(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Any]:
        """Pass a single prompt value to the model and stream model generations."""
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        input_args = {
            "prompt": formatted_prompt,
            # "stop_sequences": stop,
            "debug": verbose,
            "stream": True,
            "xxxx": 1,  ## Just to check the validaton
            **kwargs,
        }

        # TODO: Validation of the args is not being done!
        print(f"Replicate prediction using {self.model_name}; Args = {input_args}")

        # Wrap it in an async_generator!
        async def async_generator():
            llm_response = LLMResponse(model=self.model, prompt_tokens=0)
            prediction = replicate.predictions.create(version=self.llm, input=input_args)
            iterator = prediction.output_iterator()
            for text_chunk in iterator:
                result = llm_response.add_delta(text_chunk)
                yield result

            # Last token!
            try:
                prediction_logs = {
                    k[0]: k[1]
                    for k in (x.split(":") for x in prediction.logs.split("\n") if ":" in x)
                }
                print(f"Prediction Logs = {prediction.logs} // {prediction_logs}")
                prompt_tokens = int(prediction_logs.get("Number of tokens in prompt", 0))
                completion_tokens = int(prediction_logs.get("Number of tokens generated", 0))
                llm_response.set_token_count(prompt_tokens, completion_tokens)
            except Exception as e:
                print(f"Exception while parsing prediction logs: {e}")
            result = llm_response.get_last_delta()
            result["metrics"] = prediction.metrics
            print(f"Result = {result}")
            yield result

        return async_generator()
