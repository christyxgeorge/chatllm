from abc import ABC
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field


class LLMModelType(str, Enum):
    """The LLM Model Type"""

    CHAT_MODEL = "chat"  # Supports History, System Messages
    INSTRUCT_MOEL = "instruct"  # Supports Instructions
    TEXT_GEN_MODEL = "text_gen"  # Foundational Model


class LLMParam(BaseModel):
    name: str
    active: bool = True
    label: str
    description: Optional[str] = Field(alias="desc", default=None)
    minimum: float = Field(alias="min")
    maximum: float = Field(alias="max")
    default: float
    step: float


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
    minimum: float = Field(alias="min", default=0)
    maximum: float = Field(alias="max", default=1)
    default: float = 1
    step: float = 0.1
    label: str = "Top p"
    description: str = "Alternative to temperature sampling, nucleus sampling"


class TopK(LLMParam):
    name: str = "top_k"
    minimum: int = Field(alias="min", default=1)
    maximum: int = Field(alias="max", default=1000)
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

    mtype: LLMModelType = LLMModelType.CHAT_MODEL
    """Model Type: chat, instruct, text_gen"""

    max_context_length: int = Field(alias="ctx", default=2048)
    """Context Length"""

    cost_per_token: float = Field(alias="cpt", default=0.0)
    """Cost per token"""

    # LLM Parameters
    max_tokens: LLMParam = MaxTokens()
    temperature: LLMParam = Temperature()
    top_p: LLMParam = TopP()
    top_k: LLMParam = TopK()
    length_penalty: LLMParam = LengthPenalty()
    repeat_penalty: LLMParam = RepeatPenalty()
    num_sequences: LLMParam = NumSequences()

    def get_params(self) -> Dict[str, LLMParam]:
        """Return Parameters supported by the model"""
        params = {
            k: getattr(self, k)
            for k, v in self.__class__.__fields__.items()  # type: ignore
            if v.annotation == LLMParam and getattr(self, k).active
        }
        return params
