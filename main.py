import argparse
import logging
import os
import warnings
from typing import Literal

from dotenv import dotenv_values, load_dotenv

# ===========================================================================================
# Main Logic. Setup Environment and Start UI
# ===========================================================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shell", default=False, action="store_true", help="using shell mode")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--concurrency-count", type=int, default=75)
    parser.add_argument("--verbose", action="store_true", help="using verbose mode")
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    args = parser.parse_args()
    print(f"Arguments = {args}")
    return args


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


def initialize_config(verbose=False, debug=False):
    # ===========================================================================================
    # Logging Setup
    # Django method = logging.config.dictConfig(config)
    # ===========================================================================================
    log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
    log_style: Literal["%", "{", "$"] = "{"
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)
    logging.basicConfig(format=log_format, level=log_level, datefmt="%I:%M:%S", style=log_style)


def gradio_app(args):
    from grapp import setup_gradio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        demo = setup_gradio()

    # Launch Gradio App with Queuing to support streaming!
    demo.queue(
        concurrency_count=args.concurrency_count,
        max_size=100,
        status_update_rate=10,
        # api_open=False
    ).launch(server_name=args.host, server_port=args.port, debug=args.debug)


if __name__ == "__main__":
    args = parse_args()
    set_env(debug=args.debug)
    initialize_config(verbose=args.verbose, debug=args.debug)

    # Start Gradio App
    gradio_app(args)
