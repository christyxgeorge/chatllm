import logging
import random
from typing import Any

import pytest
from chatllm.llm_controller import LLMController
from chatllm.prompts import PromptValue
from tests.conftest import BATCH_TEST_PAIRS

logger = logging.getLogger(__name__)

# ====================================================================================================
# To run one test
# poetry run pytest -v tests/test_llm.py::TestLLMGeneration::test_hugging_face_batch
# To run one test with logs
# poetry run pytest -v --log-cli-level=INFO tests/test_llm.py::TestLLMGeneration::test_replicate_batch
# ====================================================================================================


pytest_plugins = ("pytest_asyncio",)

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
EXAMPLES = [
    ["Hello there! How are you doing?"],
    ["Can you explain to me briefly what is Python programming language?"],
    # [
    #     "please write a python program to find the first n numbers of the fibonacci series, where n is the input variable"
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
    #     "In bash, how do i list all the text files in the current directory that have been modified in the last month"
    # ],
    ["Name the planets in the solar system?"],
]


class TestLLMGeneration:
    def test_check(self, request, pytestconfig, batch_pairs) -> None:
        mode = pytestconfig.getoption("mode")  # pytestconfig = request.config
        print(f"Checking:")
        print(f"   Root Dir = {request.config.rootdir}")
        print(f"   Batch Pairs = {batch_pairs}")
        print(f"   Mode = {mode} // {request.config.getoption('mode')}")

    def _initialize_llm_controller(
        self, mode, provider, model_name=None, prompt=None, **kwargs
    ) -> tuple[
        LLMController, PromptValue, dict[Any, Any]
    ]:  # -> tuple[LLMController, PromptValue, dict[Any, Any]]:
        llm_controller = LLMController()
        if not model_name:
            supported_models = llm_controller.get_provider_model_list(provider)
            assert supported_models, "No models found"  # nosec
            model_name = random.choice(supported_models)  # nosec
        else:
            model_name = f"{provider}:{model_name}"
        logger.info(f"Running [{mode}] test for {provider}, model = {model_name}")
        llm_controller.load_model(model_name)
        prompt = prompt or random.choice(EXAMPLES)[0]  # nosec
        prompt_value = llm_controller.create_prompt_value(prompt, SYSTEM_PROMPT, chat_history=[])
        params = llm_controller.get_model_params(model_name)
        llm_kwargs = {k: (v["default"] if isinstance(v, dict) else v) for k, v in params.items()}
        llm_kwargs.update(kwargs)
        return llm_controller, prompt_value, llm_kwargs

    async def run_llm_batch(self, provider, model_name=None, prompt=None, **kwargs):
        llm_controller, prompt_value, llm_kwargs = self._initialize_llm_controller(
            "batched", provider, model_name=model_name, prompt=prompt, **kwargs
        )
        response_type, response_text = await llm_controller.run_batch(prompt_value, **llm_kwargs)
        assert response_type not in ["error", "warning"], f"{response_text}"  # nosec

    async def run_llm_stream(self, provider, model_name=None, prompt=None, **kwargs):
        llm_controller, prompt_value, llm_kwargs = self._initialize_llm_controller(
            "streamed", provider, model_name=model_name, prompt=prompt, **kwargs
        )
        stream = llm_controller.run_stream(prompt_value, **llm_kwargs)
        async for response_type, response_text in stream:
            assert response_type not in ["error", "warning"], f"{response_text}"  # nosec

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model_name", BATCH_TEST_PAIRS)
    async def test_batch(self, provider, model_name):
        await self.run_llm_batch(provider, model_name, max_tokens=100)

    @pytest.mark.asyncio
    async def test_openai_batch(self):
        await self.run_llm_batch("openai", "gpt-3.5-turbo", max_tokens=100)

    # Tests for batch APIs
    @pytest.mark.asyncio
    async def test_llamacpp_batch(self):
        from chatllm.llms.llama_cpp import LlamaCpp

        supported_models = [m for m in LlamaCpp.get_supported_models() if "codellama" not in m]

        assert supported_models, "No LLamaCpp models found"  # nosec
        await self.run_llm_batch("llama-cpp", supported_models[0], max_tokens=100)

    @pytest.mark.asyncio
    async def test_replicate_batch(self):
        await self.run_llm_batch("replicate", "replicate/vicuna-13b", max_tokens=100)

    @pytest.mark.asyncio
    async def test_hugging_face_batch_ts(self):
        prompt = "There was a girl called Lily and she"
        await self.run_llm_batch("hf", "roneneldan/TinyStories-33M", max_tokens=100, prompt=prompt)

    @pytest.mark.asyncio
    async def test_hugging_face_batch_g2(self):
        prompt = "There was a girl called Lily and she"
        await self.run_llm_batch("hf", "gpt2", max_tokens=100, prompt=prompt)

    # Test Streaming APIs
    @pytest.mark.asyncio
    async def test_openai_stream(self, request):
        print(f"Test OpenAI Streaming API, Root Dir = {request.config.rootdir}")
        await self.run_llm_stream("openai", "gpt-3.5-turbo", max_tokens=100)

    @pytest.mark.asyncio
    async def test_llamacpp_stream(self):
        from chatllm.llms.llama_cpp import LlamaCpp

        supported_models = [m for m in LlamaCpp.get_supported_models() if "codellama" not in m]

        assert supported_models, "No LLamaCpp models found"  # nosec
        await self.run_llm_stream("llama-cpp", supported_models[0], max_tokens=100)

    @pytest.mark.asyncio
    async def test_replicate_stream(self):
        await self.run_llm_stream("replicate", "replicate/vicuna-13b", max_tokens=100)

    @pytest.mark.asyncio
    async def test_hugging_face_stream(self):
        await self.run_llm_stream("hf", "roneneldan/TinyStories-33M", max_tokens=100)
