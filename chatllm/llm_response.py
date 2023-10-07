"""The LLM Response class. (modeled from the OpenAI response)"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, model_validator

# TODO: ID, is it per response or once at the constructor
# TODO: result_object -> do it more intelligently... based on the model provider!

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """The LLM Response class. (modeled from the OpenAI response)"""

    model: str
    """The name of the model."""

    unique_id: str
    result_object: str = "cllm.generation"
    """The unique id of the response."""

    usage: bool = True
    """Whether to include usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    """
    The number of tokens in the prompt and completion respectively.
    Note: If more than one sequence is generated, the completion tokens will 
    be the sum of all the  sequences.
    """

    @model_validator(mode="before")
    def set_default_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["unique_id"] = f"cllm-{random.randint(10000000,99999999):08d}"
        return values

    def get_result(self, sequences: str | List[str]):
        seq_list = sequences if isinstance(sequences, list) else [sequences]
        choices = [
            {"message": {"role": "assistant", "content": seq}, "finish_reason": None, "index": i}
            for i, seq in enumerate(seq_list)
        ]
        return {
            "id": self.unique_id,
            "object": self.result_object,
            "created": int(datetime.timestamp(datetime.now())),
            "model": self.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }

    def add_delta(self, delta: str | List[str]):
        """Add a delta to the response"""

        delta_list = delta if isinstance(delta, list) else [delta]
        choices = [
            {"delta": {"role": "assistant", "content": d}, "finish_reason": None, "index": i}
            for i, d in enumerate(delta_list)
        ]
        self.completion_tokens += len(delta)  # One token per completion!
        return {
            "id": self.unique_id,
            "object": self.result_object,
            "created": int(datetime.timestamp(datetime.now())),
            "model": self.model,
            "choices": choices,
        }

    def get_last_delta(self, num_deltas=1):
        """We may be generating multiple number of responses for a single prompt"""
        choices = [
            {"delta": {"role": "assistant", "content": ""}, "finish_reason": "stop", "index": i}
            for i in range(num_deltas)
        ]
        return {
            "id": self.unique_id,
            "object": self.result_object,
            "created": int(datetime.timestamp(datetime.now())),
            "model": self.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        }

    def set_token_count(self, prompt_count, completion_count) -> None:
        if self.prompt_tokens > 0 and self.prompt_tokens != prompt_count:
            logger.info(
                f"Prompt Token Count {prompt_count} is different from the computed value: {self.prompt_tokens}"
            )
        if self.completion_tokens > 0 and self.completion_tokens != completion_count:
            logger.info(
                f"Completion Token Count {completion_count} is different from the computed value: {self.completion_tokens}"
            )
        self.prompt_tokens = prompt_count
        self.completion_tokens = completion_count
