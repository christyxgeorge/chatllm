import argparse
import copy
import logging
import sys
import warnings

from typing import Literal

from chatllm.utils import set_env

# ===========================================================================================
# Main Logic. Setup Environment and Start UI
# ===========================================================================================


def parse_args(cmdline_args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chatllm", description="Chat LLM")
    subparsers = parser.add_subparsers(title="Gradio Options", dest="command", required=False)

    common_parser = argparse.ArgumentParser(add_help=False)
    # common_parser.add_argument(
    #     "-m",
    #     "--model_id",
    #     choices=BaseLanguageModel.models(),
    #     default=BaseLanguageModel.default_model(),
    # )
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False)
    common_parser.add_argument(
        "--debug", action="store_true", default=False, help="using debug mode"
    )

    grad_parser = subparsers.add_parser("gradio", parents=[common_parser])
    grad_parser.add_argument("--host", type=str, default="127.0.0.1")
    grad_parser.add_argument("-", "--port", type=int, default=7860)
    grad_parser.add_argument("--concurrency-count", type=int, default=75)

    shell_parser = subparsers.add_parser("shell", parents=[common_parser])  # noqa: F841

    args, extra_args = parser.parse_known_args(cmdline_args)
    if not args.command:
        args = common_parser.parse_args(extra_args, args)
        args.command = "shell"

    if args.verbose:
        print(f"Arguments = {args}")  # noqa: T201
    return args


def initialize_logging(verbose=False, debug=False) -> None:
    # ===========================================================================================
    # Logging Setup
    # Django method = logging.config.dictConfig(config)
    # ===========================================================================================
    log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
    log_style: Literal["%", "{", "$"] = "{"
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARN)
    logging.basicConfig(format=log_format, level=log_level, datefmt="%I:%M:%S", style=log_style)


def gradio_app(args) -> None:
    from grapp import setup_gradio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        demo = setup_gradio(verbose=args.verbose)

    # Launch Gradio App with Queuing to support streaming!
    demo.queue(
        concurrency_count=args.concurrency_count,
        max_size=100,
        status_update_rate=10,
        # api_open=False
    ).launch(server_name=args.host, server_port=args.port, debug=args.debug)


if __name__ == "__main__":
    # print(f"Locals = {sys.argv}")
    debug = "--debug" in sys.argv or "-d" in sys.argv
    set_env(debug=debug)

    cmdline_args = copy.deepcopy(sys.argv)
    args = parse_args(cmdline_args[1:])  # Exclude program name!
    initialize_logging(verbose=args.verbose, debug=args.debug)

    if args.command == "shell":
        from cli import cli

        # TODO: Need to fix this hack and make CLI command execution possible!
        if "shell" not in sys.argv:
            sys.argv.insert(1, "shell")

        cli()  # pylint: disable=no-value-for-parameter
    else:
        # Start Gradio App
        gradio_app(args)
