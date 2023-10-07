"""Prompt Classes"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from .chat_message import ChatMessage, ChatRole


class PromptValue(BaseModel, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    @abstractmethod
    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""

    def __len__(self) -> int:
        """Length of the Prompt Value"""
        chars = self.to_string()
        return len(chars)

    def get_prompt(self, role: ChatRole = ChatRole.USER) -> str:
        """Return prompt string based on the role"""
        # TODO: Do we need this?


class StringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    """Prompt text."""

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.text

    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""
        return [{"role": "user", "content": self.text}]

    def __repr__(self) -> str:
        """Represent the String Prompt Value"""
        formatted_rep = f"""Prompt: 
           {self.text}
        """
        return formatted_rep


class ChatPromptValue(PromptValue):
    """Chat prompt value."""

    messages: List[ChatMessage] = []
    """List of messages."""

    def add_message(self, msg: ChatMessage):
        """Add a message to the prompt"""
        self.messages.append(msg)

    def to_string(self) -> str:
        """Return prompt as string."""
        return self._formatted_str()

    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

    def _formatted_str(self) -> str:
        """Represent the Chat Prompt Value"""
        string_messages = []
        for msg in self.messages:
            message = f"{msg.role.value.capitalize()}: {msg.content}"
            if msg.role == ChatRole.AI and "function_call" in msg.additional_kwargs:
                message += f"{msg.additional_kwargs['function_call']}"
            string_messages.append(message)
        return "\n".join(string_messages)

    def __repr__(self) -> str:
        return self._formatted_str()

    def __str__(self) -> str:
        formatted_repr = "[\n"
        formatted_repr += self._formatted_str()
        formatted_repr += "\n]"
        return formatted_repr
