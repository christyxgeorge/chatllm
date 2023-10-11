"""The LLM Response class. (modeled from the OpenAI response)"""

import json
import logging
import random
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """The LLM Response class. (modeled from the OpenAI response)"""

    model: str
    """The name of the model."""

    unique_id: str
    """The unique id of the response."""

    num_sequences: int = 1
    """The number of sequences generated."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    """
    The number of tokens in the prompt and completion respectively.
    Note: If more than one sequence is generated, the completion tokens will 
    be the sum of all the  sequences.
    """

    api_usage: Dict[str, Any] = {}
    """
    Usage as returned by the API
    """

    finish_reason: Optional[str] = None
    """
    The completion reason for the response.
    One of 'stop', 'length', 'function' or 'error'
    Note: In OpenAI Api, the finish_reason is for each sequence. Here, we are keeping
    across all the sequences.
    """

    start_time: datetime = datetime.now()
    first_token_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    """
    Start, End times for the generation process. Also, capture the timestamp for the 
    first token creation
    """

    created_ts: int = 0
    """The creation timestamp of the response."""

    extra_info: Dict[str, Any] = {}
    """Any extra info returned from the API (like metrics)"""

    response_sequences: List[str] = []

    @model_validator(mode="before")
    def set_default_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["unique_id"] = f"cllm-{uuid.uuid4()}"
        values["created_ts"] = int(datetime.timestamp(datetime.now()))
        # Initialize the sequences!
        values["response_sequences"] = [""] * values.get("num_sequences", 1)
        return values

    def set_response(self, message: str | List[str], finish_reason: str = "stop"):
        """Set the LLM response message, when the response is not streamed (all at once))"""

        self.response_sequences = message if isinstance(message, list) else [message]
        assert self.num_sequences == len(self.response_sequences)  # nosec
        self.finish_reason = finish_reason
        self.end_time = datetime.now()

    def set_openai_response(self, response: Dict[str, Any]):
        """Set the LLM response message, when the response is not streamed (all at once))"""
        choices = response.get("choices", [])
        response_texts = [res["message"]["content"] for res in choices]
        finish_reasons = [res["finish_reason"] for res in choices]
        finish_reason = finish_reasons[0] if finish_reasons else None
        self.set_response(response_texts, finish_reason)
        self.set_api_usage(response["usage"])

    def add_openai_delta(self, delta: Dict[str, Any]) -> None:
        """Add a LLM token to the response, when the response is streamed (OpenAI response format)"""
        choices = delta.get("choices", [])
        finish_reason = None
        if choices:
            has_content = all(["content" in res["delta"] for res in choices])
            if has_content:
                response_texts = [res["delta"].get("content", None) for res in choices]
                self.add_delta(response_texts)

        if not has_content:  # Last token!
            finish_reasons = [res["finish_reason"] for res in choices]
            finish_reason = "|".join(set(finish_reasons))
            self.add_last_delta(finish_reason=finish_reason)
            self.set_api_usage(delta.get("usage", {}))

    def add_delta(self, delta: str | List[str]) -> None:
        """Add a LLM token to the response, when the response is streamed"""

        delta_list = delta if isinstance(delta, list) else [delta]
        assert self.num_sequences == len(delta_list)  # nosec
        if not self.first_token_time:
            self.first_token_time = datetime.now()

        for i, seq in enumerate(delta_list):
            self.response_sequences[i] += seq

        self.completion_tokens += len(delta_list)  # One token per completion!

    def add_last_delta(self, finish_reason="stop"):
        """
        Add the last token, when the response is streamed.
        This is to ensure that the finish_reason is propagated back!
        """
        self.end_time = datetime.now()
        if finish_reason:
            self.finish_reason = finish_reason

    def set_token_count(self, prompt_count, completion_count) -> None:
        if self.prompt_tokens > 0 and self.prompt_tokens != prompt_count:
            logger.warning(
                f"Prompt Token Count {prompt_count} is different from the computed value: {self.prompt_tokens}"
            )
        if self.completion_tokens > 0 and self.completion_tokens != completion_count:
            logger.warning(
                f"Completion Token Count {completion_count} is different from the computed value: {self.completion_tokens}"
            )
        self.prompt_tokens = prompt_count
        self.completion_tokens = completion_count

    def set_api_usage(self, usage: Dict[str, Any]):
        self.api_usage = usage
        logger.debug(f"Setting API Usage = {usage}")
        # In case, we have not computed completion tokens, we use the api_usage!
        if self.prompt_tokens == 0:
            self.prompt_tokens = usage.get("prompt_tokens", 0)
        if self.completion_tokens == 0:
            self.completion_tokens = usage.get("completion_tokens", 0)

    def set_extra_info(self, **kwargs):
        self.extra_info.update(kwargs)

    def get_first_sequence(self):
        """Return the first sequence"""
        return self.response_sequences[0] if self.response_sequences else ""

    def print_summary(self):
        elapsed_time = (self.end_time - self.start_time).total_seconds()
        response_text = self.get_first_sequence()
        usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
        print(f"Response: {response_text}")
        print(f"    Model: {self.model}")
        print(f"    Computed Usage = {json.dumps(usage or {})}")
        if self.api_usage:
            print(f"    API Usage = {json.dumps(self.api_usage)}")
        print(f"    Stop Reason = {self.finish_reason or 'n/a'}")
        if "metrics" in self.extra_info:
            print(f"    Metrics = {json.dumps(self.extra_info['metrics'])}")
        if self.first_token_time:
            token_gen_time = (self.end_time - self.first_token_time).total_seconds()
            token_gen_str = f"Time between first token and last token: {token_gen_time:.03f} secs"
        else:
            token_gen_str = ""
        if elapsed_time > 60:
            elapsed_secs = elapsed_time % 60
            elapsed_sec_str = f"{int(elapsed_time//60)} mins, {elapsed_secs:.03f} secs)"
        else:
            elapsed_sec_str = f"{elapsed_time:.03f} secs"
        tokens_per_sec = (
            f"{(usage['completion_tokens'] / elapsed_time):.02f} Tokens/Sec"
            if usage
            else "Not Available"
        )
        print(f"Elapsed time = {elapsed_sec_str} secs, {tokens_per_sec}. {token_gen_str}")
        print("=" * 130)
