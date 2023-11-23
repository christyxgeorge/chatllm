"""Interface from Gradio/CLI/PyTest to use the LLM models."""
import functools
import json
import logging
import os

from pathlib import Path
from typing import Dict, List

from chatllm.llm_params import LLMConfig
from chatllm.llm_session import LLMSession
from chatllm.llms import PROVIDER_ORDER
from chatllm.llms.base import BaseLLMProvider
from chatllm.prompts.default_prompts import long_system_prompt, simple_system_prompt

logger = logging.getLogger(__name__)


class LLMController:
    """A chat Controller for conversational models."""

    def __init__(self, verbose=False) -> None:
        self.verbose = verbose
        self.provider_models: Dict[str, List[str]] = {}  # Array of provider to models
        self.llm_models: Dict[str, LLMConfig] = {}  # Array of models to model config
        self.load_models()
        self.session = self.create_session()

    def check_path(self, mfile: str) -> Path:
        """Check if the file exists in the path"""
        mpath = Path(mfile)
        if not mpath.is_file():
            mpath = Path(os.environ["CHATLLM_ROOT"]) / "chatllm/data" / mfile
        if not mpath.is_file():
            raise FileNotFoundError(f"Model file [{mfile}] not found")
        return mpath

    def load_models(self, mfile=None) -> None:
        """Load Models from model definition file"""
        model_file = mfile or "models.json"
        models_file = self.check_path(model_file)
        logger.info(f"Loading Models from {models_file}")
        models_loaded = 0
        with open(models_file) as f:
            models = json.load(f)
            model_keys = [llm_config.key for llm_config in self.llm_models.values()]
            for model_config in models:
                llm_config = BaseLLMProvider.llm_config(model_config)
                logger.debug(f"LLM Config = {llm_config}")

                # Add to provider map and model map
                prov_models = self.provider_models.get(model_config["provider"], [])
                if llm_config.name in prov_models or llm_config.name in self.llm_models:
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
                    self.provider_models[model_config["provider"]] = prov_models
                    self.llm_models[llm_config.name] = llm_config
                    model_keys.append(llm_config.key)

            logger.info(
                f"Loaded {models_loaded} out of {len(models)} models in file "
                f"[Total: {len(self.llm_models)} models]"
            )

    def get_model_key_map(self) -> Dict[str, str]:
        """Return model key to model name mapping (for use in CLI)"""
        model_key_map = {
            model_cfg.key: model_name
            for model_name, model_cfg in self.llm_models.items()
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
        models = self.llm_models.keys()
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
        return self.provider_models.get(provider, [])

    def change_model(self, model=None) -> LLMSession:
        """Change the model, Create new session"""
        if not self.session or self.session.llm.model_name != model:
            self.session = self.create_session(model)
        else:
            logger.info(f"Model {model} already loaded, clearing history")
            self.session.clear_history()
        return self.session

    def create_session(self, model=None) -> LLMSession:
        """Load the model and create a new session"""
        self.model_name = model or self.get_model_list()[0]
        llm_cfg = self.llm_models.get(self.model_name)
        provider, model_name = self.model_name.split(":")
        provider_class = BaseLLMProvider.provider_class(provider)
        llm = provider_class(model_name=model_name, model_cfg=llm_cfg)
        session = LLMSession(llm, model_name, llm_cfg)
        # asyncio.run(self.llm.load())
        return session

    def get_system_prompt_list(self) -> Dict[str, str]:
        """Return the list of system prompts"""
        # TODO: Can we move this to prompts?
        return {
            "simple": simple_system_prompt,
            "long": long_system_prompt,
            "none": "",
            "custom": "",
        }

    def clear_history(self) -> None:
        """Clear the history"""
        self.session.clear_history()
