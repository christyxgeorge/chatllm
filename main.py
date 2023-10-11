import argparse
import logging
import sys
import warnings
from typing import Literal

from chatllm.utils import set_env

# ===========================================================================================
# Main Logic. Setup Environment and Start UI
# ===========================================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chatllm", description="Chat LLM")
    subparsers = parser.add_subparsers(
        title="Gradio Options", dest="command", required=False
    )
    grad_parser = subparsers.add_parser("gradio")

    grad_parser.add_argument("--host", type=str, default="127.0.0.1")
    grad_parser.add_argument("-", "--port", type=int, default=7860)
    grad_parser.add_argument("--concurrency-count", type=int, default=75)
    grad_parser.add_argument(
        "-v", "--verbose", action="store_true", help="using verbose mode"
    )
    grad_parser.add_argument("--debug", action="store_true", help="using debug mode")
    args = parser.parse_args()
    if args.verbose:
        print(f"Arguments = {args}")  # noqa: T201
    return args


def initialize_config(verbose=False, debug=False):
    # ===========================================================================================
    # Logging Setup
    # Django method = logging.config.dictConfig(config)
    # ===========================================================================================
    log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
    log_style: Literal["%", "{", "$"] = "{"  # type: ignore
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)
    logging.basicConfig(
        format=log_format, level=log_level, datefmt="%I:%M:%S", style=log_style
    )


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
    # print(f"Locals = {sys.argv}")
    if len(sys.argv) == 1:
        # No arguments passed. Add shell as default [Easier to handle as no args needed for shell!]
        sys.argv.append("shell")
    command = "gradio" if sys.argv[1] == "gradio" else "shell"
    debug = "--debug" in sys.argv or "-d" in sys.argv

    set_env(debug=debug)

    if command == "shell":
        from cli import cli

        cli()  # pylint: disable=no-value-for-parameter
    else:
        args = parse_args()
        initialize_config(verbose=args.verbose, debug=args.debug)
        # Start Gradio App
        gradio_app(args)
