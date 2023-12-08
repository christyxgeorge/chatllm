from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_validator,
)


class LLMModelType(str, Enum):
    """The LLM Model Type"""

    CHAT_MODEL = "chat"  # Supports Chatting with History, System Messages
    INSTRUCT_MODEL = "instruct"  # Supports Instructions
    TEXT_GEN_MODEL = "text_gen"  # Foundational Text Generation Model
    CODE_GEN_MODEL = "code_gen"  # Code Generation Model
    CODE_CHAT_MODEL = "code_chat"  # Code Chatting Model

    @property
    def is_chat_model(model) -> bool:
        return model in [LLMModelType.CHAT_MODEL, LLMModelType.CODE_CHAT_MODEL]


class LLMParam(BaseModel):
    name: str
    active: bool = True
    label: str
    description: Optional[str] = Field(alias="desc", default=None)
    minimum: float = Field(alias="min")
    maximum: float = Field(alias="max")
    default: float
    step: float

    @staticmethod
    def get_param_values(param: LLMParam | None) -> Dict[str, float | int]:
        """
        Utility function to create a dict of param values
        for use in Gradio App and testing
        """
        if param:
            kwargs = param.dict(exclude={"name", "active"})  # type: ignore
            kwargs["info"] = kwargs.pop("description", "")
            kwargs["value"] = kwargs["default"]
            kwargs["visible"] = True
            kwargs.pop("default", None)
        else:
            kwargs = {
                "value": 0,
                "visible": False,
            }
        return kwargs

    def update_values(self, **kwargs: Any) -> None:
        """Update the values of the param"""
        for k, v in kwargs.items():
            setattr(self, k, v)


class MaxTokens(LLMParam):
    name: str = "max_tokens"
    minimum: int = Field(alias="min", default=0)
    maximum: int = Field(alias="max", default=5000)
    default: int = 100
    step: int = 50
    label: str = "Max output tokens"
    description: str = "The maximum numbers of new tokens"


class Temperature(LLMParam):
    name: str = "temperature"
    minimum: float = Field(alias="min", default=0.01)
    maximum: float = Field(alias="max", default=1)
    default: float = 0.7
    step: float = 0.1
    label: str = "Temperature"
    description: str = "Higher values produce more diverse outputs"


class TopP(LLMParam):
    name: str = "top_p"
    minimum: float = Field(alias="min", default=0.0)
    maximum: float = Field(alias="max", default=1.0)
    default: float = 1.0
    step: float = 0.1
    label: str = "Top p"
    description: str = "Alternative to temperature sampling, nucleus sampling"


class TopK(LLMParam):
    name: str = "top_k"
    minimum: int = Field(alias="min", default=1)
    maximum: int = Field(alias="max", default=100)
    default: int = 50
    step: int = 25
    label: str = "Top k"
    description: str = "Sample from the k most likely next tokens"


class LengthPenalty(LLMParam):
    name: str = "length_penalty"
    minimum: float = Field(alias="min", default=1)
    maximum: float = Field(alias="max", default=5)
    default: float = 1
    step: float = 0.1
    label: str = "Length Penalty"


class RepeatPenalty(LLMParam):
    name: str = "repeat_penalty"
    minimum: float = Field(alias="min", default=1)
    maximum: float = Field(alias="max", default=2)
    default: float = 1.1
    step: float = 0.1
    label: str = "Repeat Penalty"


class PresencePenalty(LLMParam):
    name: str = "presence_penalty"
    minimum: float = Field(alias="min", default=-2.0)
    maximum: float = Field(alias="max", default=2.0)
    default: float = 0
    step: float = 0.1
    label: str = "Presence Penalty"


class NumSequences(LLMParam):
    name: str = "num_sequences"
    minimum: float = Field(alias="min", default=1)
    maximum: float = Field(alias="max", default=5)
    default: float = 1
    step: float = 1
    label: str = "Number of Sequences"
    description: str = "Generate 'n' Sequences"


class LLMConfig(BaseModel, ABC):
    """Clas to describe a LLM Model"""

    name: str
    description: str | None = Field(alias="desc", default=None)
    """Model Name and Description"""

    key: str | None = None
    """Short Code used in the CLI"""

    mtype: LLMModelType = LLMModelType.CHAT_MODEL
    """Model Type: chat, instruct, text_gen"""

    max_context_length: int = Field(alias="ctx", default=2048)
    """Context Length"""

    billing_model: str = "token"  # "token" or "char" (for PaLM2)
    """Billing Model - By default per token. PaLM2 uses character based billing"""

    cost_per_token: float = Field(alias="cpt", default=0.0)
    """Cost per token/character"""

    # LLM Parameters
    max_tokens: LLMParam = MaxTokens()
    temperature: LLMParam = Temperature()
    top_p: LLMParam = TopP()
    top_k: LLMParam = TopK()
    length_penalty: LLMParam = LengthPenalty()
    repeat_penalty: LLMParam = RepeatPenalty()
    num_sequences: LLMParam = NumSequences()

    # TODO: Need to check if there is a better way to do this!
    _overrides: Any = None
    """Store the over_ride info to set values after LLMParam object creation"""

    @model_validator(mode="wrap")
    def validate_config(
        cls, values: Dict[str, Any], handler: ValidatorFunctionWrapHandler, _info: ValidationInfo
    ) -> "LLMConfig":
        """Validate the config"""
        if "provider" in values:
            values["name"] = f"{values['provider']}:{values['name']}"
            # self.pop("provider", None)
        if "model_type" in values:
            values["mtype"] = LLMModelType(values["model_type"])

        validated_self = handler(values)

        # Setup Param Over-rides
        for param in values.get("params", []):
            pvalue = values["params"][param]
            param_attr = getattr(validated_self, param, None)
            if param_attr:
                if pvalue:
                    param_attr.update_values(**pvalue)
                else:
                    param_attr.active = False
        return validated_self

    @classmethod
    def create_config(cls, model_config):
        return cls(**model_config)

    def get_params(self) -> Dict[str, LLMParam]:
        """Return Parameters supported by the model"""
        params = {
            k: getattr(self, k)
            for k, v in self.__class__.__fields__.items()  # type: ignore
            if v.annotation == LLMParam and getattr(self, k).active
        }
        return params
