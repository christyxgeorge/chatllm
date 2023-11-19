import logging
import random

from typing import Any, Tuple, cast

import pytest

from chatllm.llm_controller import LLMController
from chatllm.prompts.default_prompts import simple_system_prompt

logger = logging.getLogger(__name__)

# ====================================================================================================
# To run one test
# poetry run pytest -v \
#      tests/test_llm.py::TestLLMGeneration::test_hugging_face_batch
# To run one test with logs
# poetry run pytest -v \
#      --log-cli-level=INFO tests/test_llm.py::TestLLMGeneration::test_replicate_batch
# To run pytest with logs and with pdb (-s)
# poetry run pytest -v -s \
#      --log-cli-level=INFO tests/test_llm.py::TestLLMGeneration::test_replicate_batch
# ====================================================================================================


pytest_plugins = ("pytest_asyncio",)


SYSTEM_PROMPT = simple_system_prompt
EXAMPLES = [
    ["Hello there! How are you doing?"],
    ["Can you explain to me briefly what is Python programming language?"],
    # [
    #     """please write a python program to find the first n numbers of the fibonacci series, \
    #        where n is the input variable"""
    # ],
    ["Explain the plot of Cinderella in a sentence."],
    ["How many hours does it take a man to eat a Helicopter?"],
    ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ["Write a python function to derive the fibonacci sequence for a given input?"],
    ["What are the capitals of Mozambique and Tanzania?"],
    ["Who is the current world champion in cricket"],
    ["Sally has two brothers and two sisters. How many sisters does sally's brother have?"],
    ["Where is India"],
    # [
    #     """In bash, how do i list all the text files in the current directory \
    #        that have been modified in the last month"""
    # ],
    ["Name the planets in the solar system?"],
]


class TestLLMGeneration:
    def test_check(self, request, pytestconfig) -> None:
        logger.info("Checking:")
        logger.info(f"   Root Dir = {request.config.rootdir}")
        params = ["mode", "provider", "num_models"]
        for param in params:
            pvalue = pytestconfig.getoption(param)  # pytestconfig = request.config
            logger.info(f"   {param.replace('_', ' ').capitalize()} = {pvalue}")

    def _setup_prompt_params(
        self, llm_controller, mode, provider_model, prompt=None, **kwargs
    ) -> Tuple[str, dict[Any, Any]]:
        llm_controller.change_model(provider_model)
        params = llm_controller.session.get_model_params()
        llm_kwargs = {k: v.default for k, v in params.items()}
        llm_kwargs.update(kwargs)

        provider, model = provider_model.split(":", maxsplit=2)
        logger.info(f"Running [{mode}] test for {provider}, model = {model}")
        user_query = prompt or random.choice(EXAMPLES)[0]  # nosec
        return user_query, llm_kwargs

    @pytest.mark.asyncio
    async def test_batch(self, request, provider_model) -> None:
        if request.config.getoption("mode") == "stream":
            pytest.mark.skip("Skipping batch test")
        else:
            llm_controller: LLMController = cast(LLMController, pytest.llm_controller)
            user_query, llm_kwargs = self._setup_prompt_params(
                llm_controller, "batched", provider_model, max_tokens=100
            )
            stream = llm_controller.session.run_batch(user_query, **llm_kwargs)
            async for response_type, response_text in stream:
                assert response_type not in [
                    "error",
                    "warning",
                ], f"{response_text}"  # nosec
                if response_type != "done":
                    logger.info(f"Response [{response_type}]: {response_text}")

    @pytest.mark.asyncio
    async def test_stream(self, request, provider_model) -> None:
        if request.config.getoption("mode") == "batch":
            pytest.mark.skip("Skipping stream test")
        else:
            llm_controller: LLMController = cast(LLMController, pytest.llm_controller)
            user_query, llm_kwargs = self._setup_prompt_params(
                llm_controller, "streamed", provider_model, max_tokens=100
            )
            stream = llm_controller.session.run_stream(user_query, **llm_kwargs)
            async for response_type, response_text in stream:  # type:ignore
                assert response_type not in [
                    "error",
                    "warning",
                ], f"{response_text}"  # nosec
