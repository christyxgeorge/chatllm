import argparse
import os

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
    args = parser.parse_args()
    return args


def set_env(verbose=False):
    """Load Environment Variables..."""

    cur_dir = os.path.abspath(os.getcwd())
    if verbose:
        config = dotenv_values(".env")
        print(f"Current directory = {cur_dir}; Dotenv Values = {config}")

    load_dotenv(".env")
    os.environ["CHATLLM_ROOT"] = cur_dir


def gradio_app(args):
    from grapp import setup_gradio

    gr_app = setup_gradio()

    # Launch Gradio App with Queuing to support streaming!
    gr_app.queue(
        concurrency_count=args.concurrency_count,
        max_size=100,
        status_update_rate=10,
        # api_open=False
    ).launch(server_name=args.host, server_port=args.port, debug=args.verbose)


if __name__ == "__main__":
    args = parse_args()
    set_env(verbose=args.verbose)

    # Start Gradio App
    gradio_app(args)
