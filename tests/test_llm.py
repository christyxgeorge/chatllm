import random
from typing import Optional

import pytest
from chatllm.llm_controller import LLMController
from chatllm.prompts import (
    ChatMessage,
    ChatPromptValue,
    ChatRole,
    PromptValue,
    StringPromptValue,
)

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
    def _create_prompt_value(self, user_query, system_prompt, chat_history=[]) -> PromptValue:
        """Create a PromptValue object"""
        prompt_value: Optional[PromptValue] = None
        if system_prompt or len(chat_history) > 1:
            prompt_value = ChatPromptValue()
            if system_prompt:
                prompt_value.add_message(ChatMessage(role=ChatRole.SYSTEM, content=system_prompt))
            for user_msg, ai_msg in chat_history:
                if user_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_msg))
                if ai_msg:
                    prompt_value.add_message(ChatMessage(role=ChatRole.AI, content=ai_msg))
            if not chat_history:
                # User Query is included in the chat history.. Add only when there is no chat_history
                prompt_value.add_message(ChatMessage(role=ChatRole.USER, content=user_query))
        else:
            prompt_value = StringPromptValue(text=user_query)
        return prompt_value

    async def run_llm_batch(self, provider, model_name, prompt=None, **kwargs):
        model_name = f"{provider}:{model_name}"
        print(f"Running test for {provider}, model = {model_name}")
        llm_controller = LLMController()
        llm_controller.load_model(model_name)
        prompt = prompt or random.choice(EXAMPLES)[0]  # nosec
        prompt_value = self._create_prompt_value(prompt, SYSTEM_PROMPT, chat_history=[])
        params = llm_controller.get_model_params(model_name)
        llm_kwargs = {k: (v["default"] if isinstance(v, dict) else v) for k, v in params.items()}
        llm_kwargs.update(kwargs)
        response_type, response_text = await llm_controller.run_batch(prompt_value, **llm_kwargs)
        assert response_type not in ["error", "warning"], f"{response_text}"  # nosec

    async def run_llm_stream(self, provider, model_name, prompt=None, **kwargs):
        model_name = f"{provider}:{model_name}"
        print(f"Running test for {provider}, model = {model_name}")
        llm_controller = LLMController()
        llm_controller.load_model(model_name)
        prompt = prompt or random.choice(EXAMPLES)[0]  # nosec
        prompt_value = self._create_prompt_value(prompt, SYSTEM_PROMPT, chat_history=[])
        params = llm_controller.get_model_params(model_name)
        llm_kwargs = {k: (v["default"] if isinstance(v, dict) else v) for k, v in params.items()}
        llm_kwargs.update(kwargs)
        stream = llm_controller.run_stream(prompt_value, **llm_kwargs)
        async for response_type, response_text in stream:
            assert response_type not in ["error", "warning"], f"{response_text}"  # nosec

    @pytest.mark.asyncio
    async def test_openai_batch(self, request):
        print(f"Test OpenAI Batch, Root Dir = {request.config.rootdir}")
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
    async def test_hugging_face_batch(self):
        await self.run_llm_batch("hf", "roneneldan/TinyStories-33M", max_tokens=100)

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
