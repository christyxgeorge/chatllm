"""Prompt Classes"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from .chat_message import ChatMessage, ChatRole
from .prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)


class PromptValue(BaseModel, ABC):
    """Base abstract class for inputs/prompts to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
    ChatModel inputs.
    """

    @abstractmethod
    def to_string(self, format="role:content") -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""

    def __len__(self) -> int:
        """Length of the Prompt Value"""
        chars = self.to_string()
        return len(chars)


class StringPromptValue(PromptValue):
    """String prompt value."""

    text: str
    """Prompt text."""

    def to_string(self, format="role:content") -> str:
        """Return prompt as string."""
        return PromptFormatter.format(
            [ChatMessage(role=ChatRole.USER, content=self.text)], name=format
        )

    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""
        return [{"role": "user", "content": self.text}]


class ChatPromptValue(PromptValue):
    """Chat prompt value."""

    messages: List[ChatMessage] = []
    """List of messages."""

    def add_message(self, msg: ChatMessage):
        """Add a message to the prompt"""
        self.messages.append(msg)

    def to_string(self, format="role:content") -> str:
        """Return prompt as string."""
        return PromptFormatter.format(self.messages, name=format)

    def to_messages(self) -> List[dict]:
        """Return prompt as dict of messages that can be serialized to JSON."""
        return [{"role": m.role.value, "content": m.content} for m in self.messages]

    def __str__(self) -> str:
        formatted_repr = "[\n"
        formatted_repr += PromptFormatter.format(self.messages, name="role:content")
        formatted_repr += "\n]"
        return formatted_repr

    def __repr__(self):
        return repr(self.messages)
