"""Prompt Classes"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .chat_message import ChatMessage, ChatRole

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI Assistant"
DEFAULT_PROMPT_FORMAT = "role:content"


class PromptFormatterProtocol(Protocol):
    def __call__(
        self,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> str:
        ...


class PromptFormatter:
    """Prompt Formatted Class."""

    # Static dictionary of chat formats
    _CHAT_FORMATS: Dict[str, Any] = {}

    @classmethod
    def register_prompt_format(
        cls, name: str
    ) -> Callable[..., PromptFormatterProtocol]:
        logger.info(f"Registering Prompt Format: {name}")

        def decorator(f: PromptFormatterProtocol):
            cls._CHAT_FORMATS[name] = f
            return f

        return decorator

    @classmethod
    def format(cls, messages: List[ChatMessage], name=DEFAULT_PROMPT_FORMAT):
        formatter = cls._CHAT_FORMATS.get(
            name, cls._CHAT_FORMATS[DEFAULT_PROMPT_FORMAT]
        )
        return formatter(messages)


# Utility functions for formatting prompts
def _map_roles(
    messages: List[ChatMessage], role_map: Dict[str, str]
) -> List[Tuple[str, Optional[str]]]:
    """Map the message roles."""
    output: List[Tuple[str, Optional[str]]] = []
    for message in messages:
        if message.role in role_map:
            output.append((role_map[message.role], message.content))
    return output


def _format_chatml(
    system_message: str, messages: List[Tuple[str, Optional[str]]], sep: str
) -> str:
    """Format the prompt with the chatml style."""
    ret = "" if system_message == "" else system_message + sep + "\n"
    for role, message in messages:
        if message:
            ret += role + "\n" + message + sep + "\n"
        else:
            ret += role + "\n"
    return ret


# Formatter functions.
@PromptFormatter.register_prompt_format("chatml")
def format_chatml(messages: List[ChatMessage], **kwargs: Any) -> str:
    system_template = """<|im_start|>system
{system_message}"""
    system_messages = [m for m in messages if m.role == ChatRole.SYSTEM]
    system_message = (
        system_messages[0].content if system_messages else DEFAULT_SYSTEM_MESSAGE
    )
    system_message = system_template.format(system_message=system_messages[0])
    roles = dict(user="<|im_start|>user", assistant="<|im_start|>assistant")
    _messages = _map_roles(messages, roles)
    _messages.append((roles["assistant"], None))
    return _format_chatml(system_message, _messages, sep="<|im_end|>")


@PromptFormatter.register_prompt_format("user_last")
def format_basic(messages: List[ChatMessage], **kwargs: Any) -> str:
    """Format the prompt with the basic style. Use only the last user message"""
    user_messages = [m for m in messages if m.role == ChatRole.USER]
    return user_messages[-1].content if user_messages else ""


@PromptFormatter.register_prompt_format("role:content")
def role_content(messages: List[ChatMessage], **kwargs: Any) -> str:
    """Format the prompt with the basic style. Use only the last user message"""
    string_messages = []
    for msg in messages:
        message = f"{msg.role.value.capitalize()}: {msg.content}"
        if msg.role == ChatRole.AI and "function_call" in msg.additional_kwargs:
            message += f"{msg.additional_kwargs['function_call']}"
        string_messages.append(message)
    return "\n".join(string_messages)
