"""Base schema for Documents."""

import textwrap

from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field

# NOTE: for pretty printing
TRUNCATE_LENGTH = 350
WRAP_WIDTH = 70


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class Document(BaseModel):
    """Generic class for a data document."""

    doc_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique ID of the document",
    )

    text: str = Field(default="", description="Text content of the document.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )

    def set_content(self, value: str) -> None:
        """Set the content"""
        self.text = value

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def remove_metadata(self, key: str) -> None:
        self.metadata.pop(key, None)

    def __str__(self) -> str:
        source_text_truncated = truncate_text(self.get_content().strip(), TRUNCATE_LENGTH)
        source_text_wrapped = textwrap.fill(f"Text: {source_text_truncated}\n", width=WRAP_WIDTH)
        return f"Doc ID: {self.doc_id}\n{source_text_wrapped}"
