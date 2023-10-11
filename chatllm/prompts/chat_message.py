"""ChatRole and ChatMessage"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

from pydantic import BaseModel, Field


# TODO: What if role names are specific to LLM?
class ChatRole(str, Enum):
    """Roles Assigned to Prompts"""

    USER = "user"
    AI = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class ChatMessage(BaseModel):
    """
    The Chat Message class.
    Messages are the inputs and outputs of ChatModels.
    """

    content: str
    """The string contents of the message."""

    role: ChatRole = ChatRole.USER
    """The role for the message."""

    name: Optional[str] = ""
    """
    The name of the function that was executed.
    [CG: May not be needed if we consistently use additional_kwargs]
    If we need, then we need a root_validator to check if it is not empty for ChatRole.AI
    """

    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

    def __str__(self) -> str:
        """Represent the Chat Message"""
        return f"{self.role.capitalize()}: {self.content}"

    def __repr__(self) -> str:
        """Represent the Chat Message"""
        return f"{self.role.capitalize()}: {self.content}"

    @classmethod
    def from_dict(cls, _dict: Mapping[str, Any]) -> ChatMessage:
        """Convert Dict to Message"""
        role_str = _dict["role"]
        role = ChatRole(role_str)
        if role == ChatRole.AI:
            content = _dict["content"] or ""  # OpenAI returns None for tool invocations
            if _dict.get("function_call"):
                additional_kwargs = {"function_call": dict(_dict["function_call"])}
            else:
                additional_kwargs = {}
            return ChatMessage(
                role=role, content=content, additional_kwargs=additional_kwargs
            )
        else:
            return ChatMessage(role=role, content=_dict["content"])

    @classmethod
    def convert_message_to_dict(cls, message: ChatMessage) -> dict:
        """Convert Message to Dict"""
        message_dict = {"role": message.role.value, "content": message.content}
        if message.role == ChatRole.FUNCTION:
            message_dict["name"] = message.name  # type: ignore
        elif message.role == ChatRole.AI:
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs[
                    "function_call"
                ]
        if message.role != ChatRole.FUNCTION and "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]
        return message_dict
