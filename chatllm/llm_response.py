"""The LLM Response class. (modeled from the OpenAI response)"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """The LLM Response class. (modeled from the OpenAI response)"""

    model: str | None = None
    """The name of the model."""

    unique_id: str | None = None
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

    finish_reasons: Set[str] = set()
    """
    The completion reason for the response.
    One of 'stop', 'length', 'function' or 'error'
    Note: In OpenAI Api, the finish_reason is for each sequence.
    Here, we are keeping across all the sequences.
    """

    start_time: datetime = datetime.now()
    first_token_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    """
    Start, End times for the generation process. Also, capture the
    timestamp for the first token creation
    """

    created_ts: int = 0
    """The creation timestamp of the response."""

    extra_info: Dict[str, Any] = {}
    """Any extra info returned from the API (like metrics)"""

    response_sequences: List[str] = []
    last_tokens: List[str] = [""] * num_sequences

    @model_validator(mode="before")
    def set_default_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["unique_id"] = f"cllm-{uuid.uuid4()}"
        values["created_ts"] = int(datetime.timestamp(datetime.now()))
        # Initialize the sequences!
        values["response_sequences"] = [""] * values.get("num_sequences", 1)
        return values

    def set_response(
        self, message: str | List[str], finish_reasons: List[str] = ["stop"]
    ) -> None:
        """Set the LLM response message, when the response is not streamed (all at once))"""

        self.response_sequences = message if isinstance(message, list) else [message]
        assert self.num_sequences == len(self.response_sequences)  # nosec
        self.finish_reasons |= set(finish_reasons)
        self.end_time = datetime.now()

    def set_openai_response(self, response: Dict[str, Any]) -> None:
        """Set the LLM response message, when the response is not streamed (all at once))"""
        choices = response.get("choices", [])
        response_texts = [res["message"]["content"] for res in choices]
        finish_reasons = [res["finish_reason"] for res in choices]
        self.set_response(response_texts, finish_reasons=finish_reasons)
        self.set_api_usage(response["usage"])

    def add_openai_delta(self, delta: Dict[str, Any]) -> None:
        """
        Add a LLM token to the response, when the response is streamed
        (OpenAI response format)
        """
        choices = delta.get("choices", [])
        has_content = all(["content" in res["delta"] for res in choices])
        if has_content:
            if not self.first_token_time:
                self.first_token_time = datetime.now()
            for choice in choices:
                response_index = choice["index"]
                response_text = choice["delta"].get("content", None)
                self.response_sequences[response_index] += response_text
                self.last_tokens[response_index] = response_text

            self.completion_tokens += len(choices)  # One token per completion!
        else:  # Last token [Will be executed once per sequence]
            finish_reasons = [res["finish_reason"] for res in choices]
            self.add_last_delta(finish_reasons=finish_reasons)
            self.set_api_usage(delta.get("usage", {}))

    def add_delta(self, delta: str | List[str]) -> None:
        """Add a LLM token to the response, when the response is streamed"""

        delta_list = delta if isinstance(delta, list) else [delta]
        assert self.num_sequences == len(delta_list), "Sequence Count Mismatch"  # nosec
        if not self.first_token_time:
            self.first_token_time = datetime.now()

        self.last_tokens = delta_list
        for i, seq in enumerate(delta_list):
            self.response_sequences[i] += seq

        self.completion_tokens += len(delta_list)  # One token per completion!

    def add_last_delta(self, finish_reasons=["stop"]) -> None:
        """
        Add the last token, when the response is streamed.
        This is to ensure that the finish_reason is propagated back!
        """
        self.end_time = datetime.now()
        self.finish_reasons |= set(finish_reasons)

    def set_token_count(self, prompt_count, completion_count) -> None:
        if self.prompt_tokens > 0 and self.prompt_tokens != prompt_count:
            logger.warning(
                f"Prompt Token Count {prompt_count} is different "
                "from the computed value: {self.prompt_tokens}"
            )
        if self.completion_tokens > 0 and self.completion_tokens != completion_count:
            logger.warning(
                f"Completion Token Count {completion_count} is different "
                "from the computed value: {self.completion_tokens}"
            )
        self.prompt_tokens = prompt_count
        self.completion_tokens = completion_count

    def set_api_usage(self, usage: Dict[str, Any]) -> None:
        self.api_usage = usage
        logger.debug(f"Setting API Usage = {usage}")
        # In case, we have not computed completion tokens, we use api_usage!
        if self.prompt_tokens == 0:
            self.prompt_tokens = usage.get("prompt_tokens", 0)
        if self.completion_tokens == 0:
            self.completion_tokens = usage.get("completion_tokens", 0)

    def set_extra_info(self, **kwargs) -> None:
        self.extra_info.update(kwargs)

    def get_first_of_last_token(self) -> str:
        return self.last_tokens[0] if self.last_tokens else ""

    def get_first_sequence(self) -> str:
        """Return the first sequence"""
        return self.response_sequences[0] if self.response_sequences else ""

    def print_summary(self) -> None:
        elapsed_time = (self.end_time - self.start_time).total_seconds()  # type: ignore[operator]
        response_text = self.get_first_sequence()
        usage = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
        print(f"Response: {response_text}")  # noqa: T201
        print(f"    Model: {self.model}")  # noqa: T201
        print(f"    Computed Usage = {json.dumps(usage or {})}")  # noqa: T201
        if self.api_usage:
            print(f"    API Usage = {json.dumps(self.api_usage)}")  # noqa: T201
        finish_reasons = "|".join(self.finish_reasons) if self.finish_reasons else "n/a"
        print(f"    Stop Reason = {finish_reasons}")  # noqa: T201
        if "metrics" in self.extra_info:
            metrics_str = f"    Metrics = {json.dumps(self.extra_info['metrics'])}"
            print(metrics_str)  # noqa: T201
        if self.first_token_time:
            tkn_gen_time = (
                self.end_time - self.first_token_time  # type: ignore[operator]
            ).total_seconds()
            tkn_gen_str = "Time between first token and last token:"  # nosec
            tkn_gen_str = f"{tkn_gen_str} {tkn_gen_time:.03f} secs"
        else:
            tkn_gen_str = ""  # nosec
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
        summary_str = f"Time Taken: {elapsed_sec_str} ({tokens_per_sec})"
        print(summary_str)  # noqa: T201
        print("=" * 130)  # noqa: T201
