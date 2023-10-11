import logging
import os

import pytest
from dotenv import dotenv_values, load_dotenv

logger = logging.getLogger(__name__)

BATCH_TEST_PAIRS = [("openai", "gpt-3.5-turbo"), ("replicate", "replicate/vicuna-13b")]


def pytest_addoption(parser):
    """Add options to the pytest command line"""
    parser.addoption(
        "--mode", action="store", default="all", help="'stream', 'batch' or 'all'"
    )


def pytest_configure():
    set_env()
    logger.info(f"setting up test environment {os.environ['CHATLLM_ROOT']}")


def pytest_runtest_setup(item):
    # called for running each test in current directory
    logger.info(f"setting up test {item}")


@pytest.fixture(autouse=True, scope="session")
def batch_pairs(pytestconfig):
    mode = pytestconfig.getoption("mode")
    logger.info(f"Batch Pairs = {BATCH_TEST_PAIRS} // {mode}")
    return BATCH_TEST_PAIRS


def set_env(debug=False):
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
