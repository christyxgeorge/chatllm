"""Interface from Gradio/CLI/PyTest to use the LLM models."""
import functools
import json
import logging
import os

from typing import Any, AsyncGenerator, Dict, List, Optional

from chatllm.llm_params import LLMConfig, LLMParam
from chatllm.llms import PROVIDER_ORDER
from chatllm.llms.base import BaseLLMProvider
from chatllm.prompts import (
    ChatMessage,
    ChatPromptValue,
    ChatRole,
    PromptValue,
    StringPromptValue,
)

logger = logging.getLogger(__name__)

simple_system_prompt = """\
You are a helpful, respectful and honest assistant. \
Always answer as helpfully as possible, while being safe.\
"""
long_system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,
while being safe. Your answers should not include any harmful, unethical, racist, sexist,
toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased
and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of
answering something not correct. If you don't know the answer to a question, please don't share
false information.
"""


class LLMController:
    """A chat Controller for conversational models."""

    provider_models: Dict[str, List[str]] = {}  # Array of provider to models
    llm_models: Dict[str, LLMConfig] = {}  # Array of models to model config

    def __init__(self) -> None:
        self.model_name = None
        self.llm = None
        self.model_config: LLMConfig | None = None
        self.system_prompt_type = "simple"
        self.system_prompt = simple_system_prompt

    @staticmethod
    def load_models(mfile=None) -> None:
        """Load Models from model definition file"""
        model_file = mfile or "models.json"
        models_file = os.environ["CHATLLM_ROOT"] + "/chatllm/data/" + model_file
        logger.info(f"Loading Models from {models_file}")
        models_loaded = 0
        with open(models_file) as f:
            models = json.load(f)
            model_keys = [llm_config.key for llm_config in LLMController.llm_models.values()]
            for model_config in models:
                provider_class = BaseLLMProvider.provider_class(model_config["provider"])
                llm_config = provider_class.model_config(model_config)
                logger.debug(f"LLM Config = {llm_config}")

                # Add to provider map and model map
                prov_models = LLMController.provider_models.get(model_config["provider"], [])
                if llm_config.name in prov_models or llm_config.name in LLMController.llm_models:
                    logger.warning(
                        f"Duplicate model {llm_config.name} for provider {model_config['provider']}"
                    )
                elif llm_config.key and llm_config.key in model_keys:
                    logger.warning(
                        f"Duplicate model key {llm_config.key} for provider {model_config['provider']}"
                    )
                else:
                    models_loaded += 1
                    prov_models.append(llm_config.name)
                    LLMController.provider_models[model_config["provider"]] = prov_models
                    LLMController.llm_models[llm_config.name] = llm_config
                    model_keys.append(llm_config.key)

            logger.info(
                f"Loaded {models_loaded} out of {len(models)} models in file "
                f"[Total: {len(LLMController.llm_models)}]"
            )

    @staticmethod
    def get_model_key_map() -> Dict[str, str]:
        """Return model key to model name mapping (for use in CLI)"""
        if not LLMController.llm_models:
            LLMController.load_models()
        model_key_map = {
            model_cfg.key: model_name
            for model_name, model_cfg in LLMController.llm_models.items()
            if model_cfg.key
        }
        return model_key_map

    def sortby_provider(self, x, y) -> int:
        prov_x = x.split(":")[0]
        prov_y = y.split(":")[0]
        if prov_x == prov_y:
            return 1 if x > y else (0 if x == y else -1)
        else:
            idx_x = (
                PROVIDER_ORDER.index(prov_x) if prov_x in PROVIDER_ORDER else len(PROVIDER_ORDER)
            )
            idx_y = (
                PROVIDER_ORDER.index(prov_y) if prov_y in PROVIDER_ORDER else len(PROVIDER_ORDER)
            )
            return idx_x - idx_y

    def get_model_list(self) -> List[str]:
        """Return the sorted list of models"""
        if not LLMController.llm_models:
            LLMController.load_models()
        models = LLMController.llm_models.keys()
        sorted_models = sorted(models, key=functools.cmp_to_key(self.sortby_provider))
        return sorted_models

    def supported_model_list(self) -> Dict[str, List[str]]:
        """Return the sorted list of models"""
        providers = BaseLLMProvider.llm_providers.keys()
        model_map = {}
        for provider in providers:
            provider_class = BaseLLMProvider.provider_class(provider)
            model_names = provider_class.get_supported_models()
            model_map[provider] = model_names
        return model_map

    def provider_model_list(self, provider) -> List[str]:
        """return the list of models for the specified provider"""
        return LLMController.provider_models.get(provider, [])

    def load_model(self, model=None):
        """Load the model"""
        self.model_name = model or self.get_model_list()[0]
        llm_cfg = LLMController.llm_models.get(self.model_name)
        provider, model_name = self.model_name.split(":")
        provider_class = BaseLLMProvider.provider_class(provider)
        self.llm = provider_class(model_name=model_name, model_cfg=llm_cfg)
        self.model_config = llm_cfg
        # asyncio.run(self.llm.load())

    def get_model_params(self) -> Dict[str, LLMParam]:
        assert self.model_config is not None, f"Model {self.model_name} not loaded"  # nosec
        return self.model_config.get_params()

    def get_system_prompt_list(self) -> Dict[str, str]:
        """return the list of system prompts"""
        return {
            "simple": simple_system_prompt,
            "long": long_system_prompt,
            "none": "",
            "custom": "",
        }

    def set_system_prompt(self, type: str, prompt: str = "") -> None:
        """Set the system prompt"""
        if prompt:
            self.system_prompt_type = type
            self.system_prompt = prompt
        else:
            self.system_prompt_type = "none"
            self.system_prompt = ""

    def create_prompt_value(self, user_query, chat_history=[]) -> PromptValue:
        """Create a PromptValue object"""
        prompt_value: Optional[PromptValue] = None
        if self.system_prompt or len(chat_history) > 1:
            prompt_value = ChatPromptValue()
            if self.system_prompt:
                prompt_value.add_message(
                    ChatMessage(role=ChatRole.SYSTEM, content=self.system_prompt)
                )
            for user_msg, ai_msg in chat_history:
                if user_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_msg))
                if ai_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.AI, content=ai_msg))
            if not chat_history:
                # User Query is included in the chat history.. Add only when there is no chat_history # noqa: E501
                prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_query))
        else:
            prompt_value = StringPromptValue(text=user_query)
        return prompt_value

    async def run_stream(
        self,
        prompt_value: PromptValue,
        verbose=False,
        word_by_word=False,
        **kwargs,
    ) -> AsyncGenerator[Any | str, Any]:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec
        if verbose:
            logger.info("=" * 50)
            logger.info(f"Prompt = {type(prompt_value)} / {prompt_value}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            stream = await self.llm.generate_stream(prompt_value, verbose=verbose, **kwargs)
            async for llm_response in stream:
                response_text = (
                    llm_response.get_first_of_last_token()
                    if word_by_word
                    else llm_response.get_first_sequence()
                )
                if response_text:
                    yield "content", response_text
                elif llm_response.finish_reasons:
                    finish_reasons = "|".join(llm_response.finish_reasons)
                    yield "warning", f"No response from LLM [Reason = {finish_reasons}]"

            yield "done", ""
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            # TODO: Can't do gradio specific in this class!
            yield "error", f"Unable to generate response [{e}]"

    async def run_batch(
        self,
        prompt_value: PromptValue,
        verbose=False,
        **kwargs,
    ) -> Any:
        assert self.llm is not None, f"Model {self.model_name} not loaded"  # nosec
        if verbose:
            logger.info("=" * 50)
            logger.info(f"Prompt = {prompt_value}")
            logger.info(f"Model: {self.llm.model_name}, Params = {json.dumps(kwargs or {})}")

        try:
            llm_response = await self.llm.generate(prompt_value, verbose=verbose, **kwargs)
            response_text = llm_response.get_first_sequence()
            if not response_text:
                yield "warning", "No response from LLM"
            else:
                yield "content", response_text
            if verbose:
                llm_response.print_summary()

        except Exception as e:
            logger.warning(f"Exception = {e}")
            yield "error", f"Unable to generate response [{e}]"
