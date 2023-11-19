"""Main Click App"""
import asyncio
import os
import re
import textwrap
import time

from typing import Dict, List, NoReturn

import click

from click_repl import repl
from click_repl.exceptions import ExitReplException
from colorama import Fore, Style
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory  # , InMemoryHistory

from chatllm.utils import set_env

MODEL_INFO: Dict[str, str] = {}


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

    def __init__(self, llm_controller, model_key, *, verbose=False):
        self.verbose = verbose
        self.llm_controller = llm_controller
        self.session = self.set_model()
        self.streaming = True

    @property
    def model_name(self) -> str | None:
        return self.llm_controller.model_name

    def set_model(self, model_key=None) -> None:
        model_name = MODEL_INFO.get(model_key, None)
        self.session = self.llm_controller.change_model(model_name)

        params = self.session.get_model_params()
        self.llm_params = {k: v.default for k, v in params.items()}
        return self.session

    def set_llm_param(self, key: str, val: float | int) -> None:
        """Set LLM Parameter by key/value"""
        if key in self.llm_params:
            self.llm_params[key] = val
        else:
            click.echo(f"Invalid Variable: {key}, Valid Options are {self.llm_params.keys()}")

    def load_models(self, modelfile: str | None = None) -> None:
        # Re-initialize Model_INFO.
        global MODEL_INFO
        self.llm_controller.load_models(modelfile)
        MODEL_INFO = self.llm_controller.get_model_key_map()
        ctx = click.get_current_context()
        for command in llm_group.list_commands(ctx):
            cli.add_command(LLM(name=command))
        self.show_model_info()

    async def llm_stream(self, user_query: str, **llm_kwargs) -> str:
        stream = self.session.run_stream(
            user_query, verbose=self.verbose, word_by_word=True, **llm_kwargs
        )
        start = True
        async for response_type, response_text in stream:
            if start:
                click.echo("Response: ", nl=False)
                start = False
            assert response_type not in [
                "error",
                "warning",
            ], f"{response_text}"  # nosec
            click.echo(response_text, nl=False)
            if response_type == "done":
                click.echo("")  # New line
        return ""

    async def llm_batch(self, user_query: str, **llm_kwargs) -> str:
        batch_gen = self.session.run_batch(user_query, verbose=self.verbose, **llm_kwargs)
        async for response_type, response_text in batch_gen:
            if response_type != "done":
                response = response_text.strip()
                click.echo(f"Response [{response_type}]:\n{response}")
        return response

    def llm_run(self, user_query: str, model_key: str | None = None, **kwargs) -> str:
        """Common Function to execute an llm run"""
        try:
            start_time = time.time()
            current_model = self.model_name
            if model_key and current_model != MODEL_INFO[model_key]:
                self.set_model(model_key)
            llm_kwargs = {**self.llm_params}  # Set with the current LLM Values
            llm_kwargs.update(**kwargs)  # Update with any over-rides specified in kwargs
            if self.streaming:
                response = asyncio.run(self.llm_stream(user_query, **llm_kwargs))
            else:
                response = asyncio.run(self.llm_batch(user_query, **llm_kwargs))
            time_taken = time.time() - start_time
            click.echo(f"== Total Time taken = {time_taken:.2f} secs")
            return response
        except Exception as exc:  # pylint: disable=broad-exception-caught
            msg = f"Error Encountered: {exc}"
            click.echo(msg)
            return msg

    def show_model_info(self) -> None:
        click.echo(Fore.GREEN + Style.BRIGHT + "Installed Models:")
        for mkey, mname in MODEL_INFO.items():
            if mname == self.model_name:
                click.echo(Fore.RED + f" ** {mkey:10s}: {mname} ==> Active" + Fore.GREEN)
            else:
                click.echo(f"    {mkey:10s}: {mname}")
        click.echo(Style.RESET_ALL)

    def show_params(self) -> None:
        params = self.session.get_model_params()
        for pkey, pval in params.items():
            click.echo(Fore.RED + f"{pkey}: {self.llm_params.get(pkey, 'N/A')}" + Fore.RESET)
            click.echo(Fore.CYAN + f"    {pval.label} [{pval.description}]" + Fore.RESET)
            click.echo(f"    Min: {pval.minimum}, Max: {pval.maximum}, Default: {pval.default}")

    def show_system_prompts(self) -> None:
        click.echo("\nSystem Prompts:\n")
        system_prompts = self.llm_controller.get_system_prompt_list()
        for ptype, prompt in system_prompts.items():
            active = "**" if ptype == self.session.system_prompt_type else "  "
            prompt = "\n".join(textwrap.wrap(prompt, subsequent_indent="    "))
            if ptype == self.session.system_prompt_type:
                click.echo(Fore.RED + f" {active} {ptype}: {prompt}\n" + Fore.RESET)
            else:
                click.echo(f" {active} {ptype}: {prompt}\n")

    def show_context_info(self) -> None:
        """Show the key ChatLLM variables"""
        mode = "stream" if self.streaming else "batch"
        click.echo("\n")
        click.echo(
            Fore.CYAN
            + Style.BRIGHT
            + "[Context]\n"
            + f"    Model = {self.model_name} [Mode: {mode}],\n"
            + f"    Parameters = {self.llm_params},\n"
            + f"    Verbose = {self.verbose}\n"
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
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.option(
    "-m",
    "--model",
    "model_key",
    type=click.Choice([k for k in MODEL_INFO.keys()]),
    help="LLM Model [default: g35 => gpt-3.5-turbo]",
    # show_choices=True,
)
def cli(ctx, model_key, verbose):
    """The ChatLLM Context"""

    if not ctx.obj:
        from chatllm.llm_controller import LLMController

        # Create LLMController
        llm_controller = LLMController(verbose=verbose)

        # Initialize Model_INFO. Needs to be done before the llm_group.list_commands is called
        global MODEL_INFO
        MODEL_INFO = llm_controller.get_model_key_map()

        for command in llm_group.list_commands(ctx):
            cli.add_command(LLM(name=command))

        # Add the context only the first time!
        ctx.obj = ChatLLMContext(llm_controller, model_key, verbose=verbose)


@cli.command()
@click.pass_context
def help(ctx) -> None:  # pylint: disable=redefined-builtin
    """Print Help String"""
    click.echo(ctx.parent.get_help())
    ctx.obj.show_context_info()


@cli.command(name="shell")
@click.pass_obj
@click.option("-v", "--verbose", is_flag=True, default=False)
@click.option("--debug/--no-debug", default=False)
def shell_start(obj, verbose, debug):
    """Start the shell"""
    click.echo("\n")
    click.echo(
        Fore.CYAN
        + Style.BRIGHT
        + "Use 'help' for help information, or ':quit'/exit to quit."
        + Style.RESET_ALL
    )
    obj.verbose = verbose
    obj.show_context_info()

    # Initialize the REPL
    def prompt_continuation(_width, _line_number, _is_soft_wrap):
        return "." * 3 + " "  # or: return [('', '.' * width)]

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
            "history": FileHistory(f"{cur_dir}/.chatllm-cli-history"),
        },
    )


@cli.command(name="exit")
def shell_exit() -> NoReturn:
    """Exit the shell"""
    raise ExitReplException()


# ===========================================================================================
# Commands to Get/Set Context Variables
# ===========================================================================================
@cli.command(name="verbose")
@click.pass_obj
def toggle_verbose(obj):
    """Toggle verbose flag"""
    obj.verbose = not obj.verbose
    obj.show_context_info()


@cli.command(name="model")
@click.pass_obj
@click.argument("model_key", default="null")
def model_key(obj, model_key):
    """List Models/Set Model"""
    if model_key == "null":
        obj.show_context_info()
    elif model_key in MODEL_INFO:
        obj.set_model(model_key)
        obj.show_context_info()
    else:
        click.echo(f"Invalid Model Key: {model_key}, Valid Options are {MODEL_INFO.keys()}")


@cli.command(name="mode")
@click.pass_obj
@click.argument("mode", default="batch")
def llm_mode(obj, mode) -> None:
    """Set Model mode ('stream' or 'batch')"""
    if mode.startswith("stream"):
        obj.streaming = True
        obj.show_context_info()
    elif mode.startswith("batch"):
        obj.streaming = False
        obj.show_context_info()
    else:
        click.echo(f"Invalid Mode: {mode}, Valid Options are 'stream' or 'batch'")


@cli.command(name="stream")
@click.pass_obj
def llm_mode_stream(obj) -> None:
    """Set Model mode to 'stream'"""
    obj.streaming = True
    obj.show_context_info()


@cli.command(name="batch")
@click.pass_obj
def llm_mode_batch(obj) -> None:
    """Set Model mode to 'batch'"""
    obj.streaming = False
    obj.show_context_info()


@cli.command(
    name="params",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_obj
def set_llm_params(obj) -> None:
    """
    Simple way to list / set multiple Parameters to send to the LLM.
    Usage: set a=b c=d e=f,g h ## e has multiple values and h is boolean
    TODO: Can use argparse.parse(shlex.split()) to handle quotes in arguments as well!
    """
    ctx = click.get_current_context()
    if not ctx.args:
        obj.show_params()
    else:
        arg_vars = [item.split("=") if "=" in item else [item, "true"] for item in ctx.args]
        for key, value in arg_vars:
            obj.set_llm_param(key, value)
        obj.show_context_info()


@cli.command(name="sprompt")
@click.pass_obj
@click.argument("prompt_type", default="null")
def set_system_prompt(obj, prompt_type):
    """
    Simple way to list / show system prompts.
    """
    if prompt_type == "null":
        obj.show_system_prompts()
    elif prompt_type in ["simple", "long", "none"]:
        obj.llm_controller.set_system_prompt(prompt_type, "")
        obj.show_system_prompts()
    else:
        click.echo(
            f"Invalid Prompt Type: {prompt_type}, Valid Options are 'simple', 'long' or 'none'"
        )


# ===========================================================================================
# LLM Commands
# ===========================================================================================


@cli.command(name="models")
@click.pass_obj
@click.argument("action", default=None, required=False)
@click.argument("modelfile", default=None, required=False)
@click.option("-p", "--provider", is_flag=True)
def model_list(obj, action, modelfile=None, provider=False):
    """List Models/Set Model"""
    click.echo("")  # Just a newline!
    if action == "add" and modelfile:
        obj.load_models(modelfile)

    elif action == "all":
        click.echo(Fore.CYAN + "Available Models by provider:" + Fore.RESET)
        provider_map = obj.llm_controller.supported_model_list()
        for prov, models in provider_map.items():
            click.echo(f"    {prov} [{len(models)}] => {models}")
    else:  # Default Action!
        if provider:
            click.echo(Fore.CYAN + "Installed Models by provider:" + Fore.RESET)
            provider_map = obj.llm_controller.provider_models
            for prov, models in provider_map.items():
                click.echo(f"    {prov} [{len(models)}] => {models}")
        else:
            obj.show_model_info()


@cli.command(
    name="q",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_obj
def start_chat(obj) -> None:
    """
    Start new chat session with the current active model and run the user query.
    """
    ctx = click.get_current_context()
    if not ctx.args:
        obj.show_params()
    else:
        prompt = " ".join(ctx.args)
        click.echo(f"Invoked Active LLM [{obj.model_name}] with prompt, {prompt}")
        return obj.llm_run(prompt)


@cli.command(
    name="c",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_obj
def continue_chat(obj) -> None:
    """
    Continue the existing chat session
    """
    ctx = click.get_current_context()
    if not ctx.args:
        obj.show_params()
    else:
        prompt = " ".join(ctx.args)
        return obj.llm_run(prompt)


class LLMGroup(click.MultiCommand):
    def list_commands(self, ctx) -> List[str]:
        return list(MODEL_INFO.keys())

    def get_command(self, ctx, name) -> click.Command:
        return LLM(name=name)


class LLM(click.Command):
    """Invoke LLM with the specified model"""

    def __init__(self, name):
        super().__init__(name)
        self.allow_extra_args = True
        self.ignore_unknown_options = True
        self.help = f"Run {MODEL_INFO.get(self.name)}"

    def invoke(self, ctx) -> None:
        """Invoke LLM"""
        # NOTE: This can be made into a decorated function as well
        if not ctx.args:
            click.echo(f"No Prompt Specified, Changing active model to {self.name}")
            ctx.obj.set_model(self.name)
        else:
            prompt = " ".join(ctx.args)
            model_key = self.name or ctx.info_name
            model_name = MODEL_INFO[model_key]
            click.echo(f"Invoked LLM [{model_name}] with prompt, {prompt} / {self.name}")
            return ctx.obj.llm_run(prompt, self.name)


@click.command(cls=LLMGroup)
def llm_group() -> None:
    """LLM Commands"""
    pass


# ===========================================================================================
# Main Logic. Setup Environment and Start Shell
# ===========================================================================================


if __name__ == "__main__":
    set_env()
    cli()  # pylint: disable=no-value-for-parameter
