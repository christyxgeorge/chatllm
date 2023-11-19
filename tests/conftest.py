import logging
import os

from typing import List, cast

import pytest

from dotenv import dotenv_values, load_dotenv

from chatllm.llm_controller import LLMController

logger = logging.getLogger(__name__)


def set_env(debug=False) -> None:
    """Load Environment Variables..."""

    cur_dir = os.path.abspath(os.getcwd())
    if debug:
        config = dotenv_values(".env")
        logger.info(f"Current directory = {cur_dir}; Dotenv Values = {config}")

    if os.path.exists(".env"):
        load_dotenv(".env")
        os.environ["CHATLLM_ROOT"] = cur_dir
    else:
        raise ValueError("Unable to load environment variables from .env file")


def pytest_addoption(parser) -> None:
    """Add options to the pytest command line"""
    parser.addoption("--mode", action="store", default="all", help="'stream', 'batch' or 'all'")
    parser.addoption("--provider", action="store", default="all", help="provider name")
    parser.addoption("--model", action="store", default="all", help="model name")
    parser.addoption(
        "--num-models", action="store", type=int, default=0, help="number of models to test"
    )


def pytest_generate_tests(metafunc) -> None:
    if "provider_model" in metafunc.fixturenames:
        provider = metafunc.config.getoption("provider")
        models: List[str] = cast(List[str], pytest.test_models)
        func_name = metafunc.function.__name__
        logger.info(
            f"[{func_name}] Parametrizing for provider [{provider}], Models = {len(models)}"
        )
        metafunc.parametrize("provider_model", models)


def pytest_configure(config) -> None:
    set_env()

    # Setup LLM Controller once for all tests
    pytest.llm_controller = LLMController()

    # Setup filtered model list
    num_models = config.getoption("num_models")
    provider = config.getoption("provider")
    model_name = config.getoption("model")
    llm_controller: LLMController = cast(LLMController, pytest.llm_controller)
    models = llm_controller.get_model_list()
    models = models if provider == "all" else [m for m in models if m.startswith(provider)]
    models = models if model_name == "all" else [m for m in models if m.endswith(":" + model_name)]

    if model_name == "all":
        # Use num_models only if model is not specified
        models = models[:num_models] if num_models else models
    pytest.test_models = models

    # TODO: Logger is not getting printed
    logger.info(f"Mode: {config.getoption('mode')}")
    logger.info(f"setting up test environment {os.environ['CHATLLM_ROOT']}")


def pytest_runtest_setup(item) -> None:
    # called for running each test in current directory
    logger.info(f"setting up test {item}")
