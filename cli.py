"""Main Click App"""
import asyncio
import os
import re
import time
from typing import Any, Dict, Tuple

import click
from chatllm.prompts.prompt_value import PromptValue
from chatllm.utils import set_env
from click_repl import repl
from click_repl.exceptions import ExitReplException
from colorama import Fore, Style
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory  # , InMemoryHistory

DEFAULT_TEMPERATURE = 0.75
MODEL_INFO: "Dict[str, str]" = {
    "g35": "openai:gpt-3.5-turbo",
    "g4": "openai:gpt-4",
    "dv": "openai:davinci",
    "lc7b": "llama-cpp:llama-2-7b-chat.Q5_K_M.gguf",
    "m7b": "llama-cpp:mistral-7b-openorca.Q5_K_M.gguf",
    "l7b": "replicate:replicate/llama-7b",
    "l13b": "replicate:replicate/llama-13b",
    "l70b": "replicate:replicate/llama-70b",
    "vicuna": "replicate:replicate/vicuna-13b",
    "g2": "hf:gpt2",
    "p15": "microsoft/phi-1_5",
    "ts": "hf:roneneldan/TinyStories-33M",
    "rcode": "hf:replit/replit-code-v1_5-3b",
}


# ===========================================================================================
# Utility Functions
# ===========================================================================================
def normalize_token(token_name: str) -> str:
    """
    As of click>=7, underscores in function names are replaced by dashes.
    To avoid the need to rename all cli functions, e.g. load_examples to
    load-examples, this function is used to convert dashes back to
    underscores.

    :param token_name: token name possibly containing dashes
    :return: token name where dashes are replaced with underscores
    """
    return token_name.replace("_", "-")


# ===========================================================================================
# Shell Context
# ===========================================================================================


class ChatLLMContext(object):
    """Chat Context"""

    def __init__(self, model_key, *, temperature=None, verbose=False):
        self.verbose = verbose
        self.mode = "batch"
        self.model_name = MODEL_INFO[model_key]
        self.vars = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 100}
        self.llm_controller = self._initialize_llm_controller()

    def _initialize_llm_controller(self):
        from chatllm.llm_controller import LLMController

        llm_controller = LLMController()
        llm_controller.load_model(self.model_name)
        return llm_controller

    def set_model(self, model_key):
        self.model_name = MODEL_INFO[model_key]
        self.llm_controller.load_model(self.model_name)

    def add_key(self, key: str, val: Any) -> None:
        """Add Variable reference"""
        self.vars[key] = val

    def get_key(self, key: str) -> Any:
        """Add Variable reference"""
        return self.vars.get(key)

    async def llm_stream(self, prompt_value: PromptValue, **llm_kwargs) -> Tuple[str, str]:
        stream = self.llm_controller.run_stream(prompt_value, word_by_word=True, **llm_kwargs)
        click.echo("Response: ", nl=False)
        async for response_type, response_text in stream:
            assert response_type not in ["error", "warning"], f"{response_text}"  # nosec
            click.echo(response_text, nl=False)
            # await asyncio.sleep(1)  # Sleep to check if this is working!
        click.echo("")  # New line
        return "content", ""

    def llm_run(self, prompt: str, model_key: str | None = None, **kwargs):
        """Common Function to execute an llm run"""
        try:
            start_time = time.time()
            if model_key and self.model_name != MODEL_INFO[model_key]:
                self.model_name = MODEL_INFO[model_key]
                self.llm_controller.load_model(self.model_name)
            params = self.llm_controller.get_model_params(self.model_name)
            llm_kwargs = {
                k: (v["default"] if isinstance(v, dict) else v) for k, v in params.items()
            }
            llm_kwargs.update(**self.vars)  # Update with the current variables
            llm_kwargs.update(**kwargs)  # Update with any over-rides specified in kwargs
            click.echo(f"Arguments to LLM: {llm_kwargs}")
            prompt_value = self.llm_controller.create_prompt_value(prompt, "", chat_history=[])
            if self.mode == "stream":
                response_type, response = asyncio.run(self.llm_stream(prompt_value, **llm_kwargs))
            else:
                response_type, response = asyncio.run(
                    self.llm_controller.run_batch(prompt_value, **llm_kwargs)
                )
                click.echo(f"Response [{response_type}]:\n{response.strip()}")
            time_taken = time.time() - start_time
            click.echo(f"== Time taken = {time_taken:.2f} secs")
            return response.strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = f"Error Encountered: {exc}"
            click.echo(msg)
            return msg

    def show_context_info(self) -> None:
        """Show the key ChatLLM variables"""
        click.echo("\n")
        click.echo(
            Fore.CYAN
            + Style.BRIGHT
            + f"[Context] Model = {self.model_name} Mode: [{self.mode}], "
            + f"Arguments = {self.vars}, "
            + f"Verbose = {self.verbose}\n"
            + Style.RESET_ALL
        )
        click.echo("\n")


# ===========================================================================================
# Shell Creation and Control Commands (Help, Exit)
# ===========================================================================================


@click.group(
    invoke_without_command=True,
    context_settings={"token_normalize_func": normalize_token},
)
@click.pass_context
@click.option("--verbose/--no-verbose", default=False)
@click.option(
    "-m",
    "--model",
    "model_key",
    default="g35",
    type=click.Choice([k for k in MODEL_INFO.keys()]),
    help="LLM Model [default: g35 => gpt-3.5-turbo]",
    show_choices=True,
)
@click.option("-t", "--temperature", type=float, help="LLM Temperature")
def cli(ctx, model_key, temperature, verbose):
    """The ChatLLM Shell"""
    if not ctx.obj:
        # Add the context only the first time!
        ctx.obj = ChatLLMContext(model_key, temperature=temperature, verbose=verbose)


@cli.command()
@click.pass_context
def help(ctx):  # pylint: disable=redefined-builtin
    """Print Help String"""
    click.echo(ctx.parent.get_help())
    ctx.obj.show_context_info()


@cli.command(name="shell")
@click.pass_obj
@click.option("--verbose/--no-verbose", default=False)
@click.option("--debug/--no-debug", default=False)
def shell_start(obj, verbose, debug):
    """Start the shell"""
    click.echo("\n")
    click.echo(
        Fore.CYAN
        + Style.BRIGHT
        + "Run ':help' for help information, or ':quit' to quit."
        + Style.RESET_ALL
    )
    obj.verbose = verbose
    obj.show_context_info()

    # Initialize the REPL
    def prompt_continuation(_width, _line_number, _is_soft_wrap):
        return "." * 3 + " "
        # Or: return [('', '.' * width)]

    @Condition
    def check_multiline() -> bool:
        """Check if we should activate multi-line based on unmatched quotes"""
        buffer = get_app().current_buffer
        unterminated_string_found = re.search(r"['\"][^\"']+$", buffer.text)
        return bool(unterminated_string_found)

    cur_dir = os.path.abspath(os.getcwd())
    repl(
        click.get_current_context(),
        prompt_kwargs={
            "message": "ChatLLM > ",
            "multiline": check_multiline,
            "prompt_continuation": prompt_continuation,
            "history": FileHistory(f"{cur_dir}/.chatllm-history"),
        },
    )


@cli.command(name="exit")
def shell_exit():
    """Exit the shell"""
    raise ExitReplException()


# ===========================================================================================
# Commands to Get/Set Context Variables
# ===========================================================================================


@cli.command(
    name="set",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_obj
def set_var(obj):
    """
    Sets Multiple Variables to send to the LLM.
    Needs pairs of strings.
    NOTE: Boolean variables or Multiple options can introduce issues!
          Boolean: -x -y -z --> Will create {x: y}
          Multiple: -x a b -y --> Will create { x: a, b: y}
    TODO: Can use argparse.parse(shlex.split()) to handle quotes in arguments as well!
    """
    ctx = click.get_current_context()
    click.echo(f"Arg Vars = {ctx.args}")
    arg_vars = [sub.lstrip("-") for item in ctx.args for sub in item.split("=")]
    arg_dict = dict(zip(arg_vars[::2], arg_vars[1::2]))
    click.echo(f"Variables: {arg_dict}")
    for key, value in arg_dict.items():
        obj.add_key(key, value)


@cli.command(name="get")
@click.pass_obj
@click.argument("var", required=True)
def get_var(obj, var):
    """Get the Variable"""
    click.echo(f"Variable: {obj.get(var)}")


@cli.command(name="vars")
@click.pass_obj
def list_vars(obj) -> None:
    """List the Variables that will be sent to the LLM"""
    var_list = obj.vars
    click.echo(f"Variables: {var_list}")


@cli.command(name="verbose")
@click.pass_obj
def toggle_verbose(obj):
    """Toggle verbose flag"""
    obj.verbose = not obj.verbose
    obj.show_context_info()


@cli.command(name="model")
@click.pass_obj
@click.argument("model_key", default="l7b")
def model_key(obj, model_key):
    """Set Model temperature"""
    if model_key in MODEL_INFO:
        obj.set_model(model_key)
        obj.show_context_info()
    else:
        click.echo(f"Invalid Model Key: {model_key}, Valid Options are {MODEL_INFO.keys()}")


@cli.command(name="mode")
@click.pass_obj
@click.argument("mode", default="batch")
def llm_mode(obj, mode):
    """Set Model mode ('stream' or 'batch')"""
    if mode.startswith("stream"):
        obj.mode = "stream"
        obj.show_context_info()
    elif mode.startswith("batch"):
        obj.mode = "batch"
        obj.show_context_info()
    else:
        click.echo(f"Invalid Mode: {mode}, Valid Options are 'stream' or 'batch'")


@cli.command(name="temperature")
@click.pass_obj
@click.argument("temperature", default=DEFAULT_TEMPERATURE, type=float)
def model_temperature(obj, temperature):
    """Set Model temperature"""
    obj.add_key("temperature", temperature)
    obj.show_context_info()


# ===========================================================================================
# LLM Commands
# ===========================================================================================


@cli.command()
@click.pass_obj
@click.argument("prompt", required=True)  # prompt="Enter Query", help="The Query for LLAMA-2")
@click.option(
    "-t",
    "--temperature",
    default=DEFAULT_TEMPERATURE,
    type=float,
    help="LLM Temperature",
    show_default=True,
)
def vicuna(obj, prompt: str, temperature: float):
    """Sends the user query to the Vicuna LLM on Replicate"""
    return obj.llm_run(prompt, "vicuna", temperature=temperature)


@cli.command()
@click.pass_obj
@click.argument("prompt", required=True)  # prompt="Enter Query", help="The Query for GPT-4")
@click.option(
    "-t",
    "--temperature",
    default=DEFAULT_TEMPERATURE,
    type=float,
    help="LLM Temperature",
    show_default=True,
)
def gpt4(obj, prompt: str, temperature: float):
    """Sends the user query to OpenAI GPT 4"""
    return obj.llm_run(prompt, "g4", temperature=temperature)


@cli.command()
@click.pass_obj
@click.argument("prompt", required=True)  # prompt="Enter Query", help="The Query for GPT-4")
@click.option(
    "-t", "--temperature", default=0.2, type=float, help="LLM Temperature", show_default=True
)
def gpt(obj, prompt: str, temperature: float):
    """Sends the user query to OpenAI GPT 3.5"""
    return obj.llm_run(prompt, "g35", temperature=temperature)


@cli.command()
@click.pass_obj
@click.argument("prompt", required=True)  # prompt="Enter Query", help="The Query for GPT-4")
@click.option(
    "-t", "--temperature", default=0.2, type=float, help="LLM Temperature", show_default=True
)
def davinci(obj, prompt: str, temperature: float):
    """Sends the user query to OpenAI Davinci"""
    return obj.llm_run(prompt, "dv", temperature=temperature)


@cli.command()
@click.pass_obj
@click.argument("prompt", required=True)  # prompt="Enter Query", help="The Query for GPT-4")
@click.option(
    "-t",
    "--temperature",
    default=DEFAULT_TEMPERATURE,
    type=float,
    help="LLM Temperature",
    show_default=True,
)
def llm(obj, prompt: str, temperature: float):
    """Sends the user query to the currently selected LLM"""
    return obj.llm_run(prompt, temperature=temperature)


# ===========================================================================================
# Main Logic. Setup Environment and Start Shell
# ===========================================================================================


if __name__ == "__main__":
    set_env()
    cli()  # pylint: disable=no-value-for-parameter
