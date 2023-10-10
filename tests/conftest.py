import os

from dotenv import dotenv_values, load_dotenv


def pytest_configure():
    set_env()
    print(f"setting up test environment {os.environ['CHATLLM_ROOT']}")


def pytest_runtest_setup(item):
    # called for running each test in current directory
    print(f"setting up test {item}")


def set_env(debug=False):
    """Load Environment Variables..."""

    cur_dir = os.path.abspath(os.getcwd())
    if debug:
        config = dotenv_values(".env")
        print(f"Current directory = {cur_dir}; Dotenv Values = {config}")

    if os.path.exists(".env"):
        load_dotenv(".env")
        os.environ["CHATLLM_ROOT"] = cur_dir
    else:
        raise ValueError("Unable to load environment variables from .env file")
