"""Language Model To Interface With Local Llama.cpp models"""
from __future__ import annotations

import logging
import os

from typing import Any, AsyncGenerator, List, Tuple, cast

import vertexai

from google.oauth2 import service_account
from vertexai.language_models import (
    ChatModel,
    CodeChatModel,
    CodeGenerationModel,
    TextGenerationModel,
)

from chatllm.llm_params import (
    LengthPenalty,
    LLMConfig,
    LLMModelType,
    LLMParam,
    MaxTokens,
    NumSequences,
    RepeatPenalty,
    Temperature,
    TopK,
    TopP,
)
from chatllm.llm_response import LLMResponse
from chatllm.llms.base import BaseLLMProvider, LLMRegister
from chatllm.prompts import PromptValue

logger = logging.getLogger(__name__)


class VertexConfig(LLMConfig):
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text
    # Handle parameter variations
    max_tokens: LLMParam = MaxTokens(
        name="max_output_tokens", min=0, max=1024, step=64, default=128
    )
    temperature: LLMParam = Temperature(min=0, max=1, default=0.75, step=0.05)
    length_penalty: LLMParam = LengthPenalty(active=False)
    repeat_penalty: LLMParam = RepeatPenalty(active=False)
    num_sequences: LLMParam = NumSequences(name="candidate_count", min=1, max=8, default=1)
    top_k: LLMParam = TopK(min=0, max=40, default=40, step=10)

    # class to be used:
    vertex_class: Any


VERTEX_MODEL_LIST: List[VertexConfig] = [
    VertexConfig(
        name="chat-bison",
        key="vcb",
        desc="PaLM-2 for Chat",
        ctx=8192,
        cpt=0.0,
        vertex_class=ChatModel,
        mtype=LLMModelType.CHAT_MODEL,
    ),
    VertexConfig(
        name="chat-bison-32k",
        desc="PaLM-2 for Chat (32K)",
        ctx=32768,
        cpt=0.0,
        vertex_class=ChatModel,
        mtype=LLMModelType.CHAT_MODEL,
    ),
    VertexConfig(
        name="text-bison",
        key="vtb",
        desc="PaLM-2 for Chat",
        ctx=8192,
        cpt=0.0,
        vertex_class=TextGenerationModel,
        mtype=LLMModelType.TEXT_GEN_MODEL,
    ),
    VertexConfig(
        name="code-bison",
        key="vcode",
        desc="Codey for code generation",
        top_p=TopP(active=False),
        top_k=TopK(active=False),
        ctx=6144,
        cpt=0.0,
        vertex_class=CodeGenerationModel,
        mtype=LLMModelType.TEXT_GEN_MODEL,
    ),
    VertexConfig(
        name="codechat-bison",
        key="vcchat",
        desc="Codey for code chat",
        top_p=TopP(active=False),
        top_k=TopK(active=False),
        ctx=6144,
        cpt=0.0,
        vertex_class=CodeChatModel,
        mtype=LLMModelType.CHAT_MODEL,
    ),
]


@LLMRegister()
class VertexApi(BaseLLMProvider):
    """Class for interfacing with GCP Vertex.AI models."""

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        project = os.environ.get("GCLOUD_PROJECT")
        location = os.environ.get("GCLOUD_LOCATION")
        staging_bucket = os.environ.get("GCLOUD_BUCKET")
        credentials_file = os.environ.get("GCLOUD_CREDENTIALS_FILE")
        credentials = service_account.Credentials.from_service_account_file(credentials_file)
        vertexai.init(
            project=project,
            location=location,
            staging_bucket=staging_bucket,
            credentials=credentials,
        )
        llm_info = [mcfg for mcfg in VERTEX_MODEL_LIST if mcfg.name == model_name][0]
        self.llm = llm_info.vertex_class.from_pretrained(model_name)
        self.model_type = llm_info.mtype

    @classmethod
    def get_supported_models(cls, verbose: bool = False) -> List[LLMConfig]:
        """Return a list of supported models."""
        return cast(List[LLMConfig], VERTEX_MODEL_LIST)

    async def load(self, **kwargs: Any) -> None:
        """
        Load the model. Nothing to do in the case of LlamaCpp
        as we load the model in the constructor.
        """
        pass

    def get_token_count(self, prompt: str) -> int:
        """Return the number of tokens in the prompt."""
        tokens: List[str] = []
        return len(tokens)

    def format_prompt(self, prompt_value: PromptValue) -> Tuple[str, int]:
        """Format the prompt for Vertex AI Predictions: Nothing to be done!"""
        formatted_prompt = prompt_value.to_string()
        return formatted_prompt, self.get_token_count(formatted_prompt)

    def validate_kwargs(self, **kwargs):
        """Validate the kwargs passed to the model"""
        # Rename max_tokens to max output tokens
        kwargs["max_output_tokens"] = kwargs.pop("max_tokens", 128)
        kwargs["candidate_count"] = kwargs.pop("num_sequences", 1)
        return kwargs

    async def generate(
        self,
        prompt_value: PromptValue,
        *,
        verbose: bool = False,
        **kwargs: Any,
    ) -> LLMResponse:
        formatted_prompt, num_tokens = self.format_prompt(prompt_value)
        validated_kwargs = self.validate_kwargs(**kwargs)
        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)

        # TODO: Check predict_async and send_message_async [Get a not awaited error]!
        if self.model_type == LLMModelType.CHAT_MODEL:
            system_prompt = prompt_value.get_system_prompt()
            chat = self.llm.start_chat(context=system_prompt)
            prediction = chat.send_message(formatted_prompt, **validated_kwargs)
        else:
            prediction = self.llm.predict(formatted_prompt, **validated_kwargs)
        llm_response.set_response(prediction.text, ["stop"])
        llm_response.set_token_count(num_tokens, 0)
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
        validated_kwargs = self.validate_kwargs(**kwargs)
        # In case of streaming, candidate count is not used as it is always one
        validated_kwargs.pop("candidate_count")

        llm_response = LLMResponse(model=self.model, prompt_tokens=num_tokens)
        if self.model_type == LLMModelType.CHAT_MODEL:
            system_prompt = prompt_value.get_system_prompt()
            chat = self.llm.start_chat(context=system_prompt)
            prediction = chat.send_message_streaming_async(formatted_prompt, **validated_kwargs)
        else:
            prediction = self.llm.predict_streaming_async(formatted_prompt, **validated_kwargs)

        # Wrap it in an async_generator!
        async def async_generator() -> AsyncGenerator[Any | str, Any]:
            async for text_chunk in prediction:
                # Note: The text_chunk can contain more than one tokens!
                llm_response.add_delta(text_chunk.text)
                yield llm_response

            # Last token!
            if llm_response.completion_tokens >= kwargs.get("max_tokens", 0):
                finish_reason = "length"
            else:
                finish_reason = "stop"
            llm_response.add_last_delta(finish_reasons=[finish_reason])
            yield llm_response

        return async_generator()
