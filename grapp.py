"""Setup the Gradio App"""
import logging

from typing import List

import gradio as gr

from chatllm.llm_controller import LLMController
from chatllm.llm_params import LLMParam

logger = logging.getLogger(__name__)

# ===========================================================================================
# Constants
# ===========================================================================================
SHARED_UI_WARNING = ""

title = "ChatLLM: A Chatbot for Language Models"
TITLE_MARKDOWN = f"""
<h1 style='text-align: center; color: #e47232'>{title}</h1>
"""


examples = [
    ["Hello there! How are you doing?"],
    ["Can you explain to me briefly what is Python programming language?"],
    [
        "please write a python program to find the first n numbers of the fibonacci series, "
        "where n is the input variable"
    ],
    ["Explain the plot of Cinderella in a sentence."],
    ["How many hours does it take a man to eat a Helicopter?"],
    ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ["Write a python function to derive the fibonacci sequence for a given input?"],
    ["What are the capitals of Mozambique and Tanzania?"],
    ["Who is the current world champion in cricket"],
    ["Sally has two brothers and two sisters. How many sisters does sally's brother have?"],
    ["Where is India"],
    [
        "In bash, how do i list all the text files in the current directory "
        "that have been modified in the last month"
    ],
    ["Name the planets in the solar system?"],
]


def load_demo(model_name, parameters: List[gr.Slider], verbose=False):
    """Load the Demo"""
    param_keys = [p.elem_id for p in parameters]
    session = llm_controller.session
    params = session.get_model_params()
    param_values = {k: LLMParam.get_param_values(params.get(k)) for k in param_keys}
    values = {k: v["value"] for k, v in param_values.items()}
    state = {
        "stream_mode": True,
        "model": model_name,
        "params": values,
        "verbose": verbose,
    }
    logger.info(f"Loaded Demo: state = {state}")
    gr_sliders = [gr.Slider(**param_values[pk]) for pk in param_keys]
    return (
        state,
        gr.Dropdown(value=model_name, visible=True),  # Model Dropdown - Make visible
        gr.Accordion(open=True, visible=True),  # parameter_row - Open and visible
        *gr_sliders,
    )


def close_parameters() -> gr.Accordion:
    """
    Close the parameters view when the model changes.
    Note: we are making visible false so that the progress information is not seen
    on the accordion
    """
    return gr.Accordion(open=False, visible=False)


# Event Handlers Implemented
def model_changed(
    state: gr.State, model_name: str, chatbot: List[str], parameters: List[gr.Slider]
):
    """Handle Model Change"""
    param_keys = [p.elem_id for p in parameters]
    param_values = {k: {"value": v} for k, v in state["params"].items()}
    if model_name != state["model"]:
        """Load Model only if the model has changed"""
        try:
            logger.info(f"Changing model from {state['model']} to {model_name}")
            llm_controller.change_model(model_name)
            params = llm_controller.session.get_model_params()
            param_values = {k: LLMParam.get_param_values(params.get(k, None)) for k in param_keys}
            values = {k: v["value"] for k, v in param_values.items()}
            state["params"] = values
            state["model"] = model_name
            chatbot = []  # Clear the chat history
        except Exception as e:
            logger.warning(f"Error loading model {model_name}: {e}")
            gr.Info(f"Unable to load model {model_name}... Using {state['model']}")
    else:
        logger.info(f"Model {model_name} has not changed")
    gr_sliders = [gr.Slider(**param_values[pk]) for pk in param_keys]
    return (
        state,
        state["model"],
        chatbot,
        gr.Accordion(open=True, visible=True),  # parameter_row - Open and visible
        *gr_sliders,
    )


def parameter_changed(state: gr.State, parameter: gr.Slider, value: int | float):
    logger.debug(f"Parameter: {parameter.elem_id} // Value: {value} / {parameter.value}")
    state["params"][parameter.elem_id] = value
    return state


def mode_changed(state: gr.State, active_mode: str):
    state.update({"stream_mode": active_mode})
    return state


def system_prompt_changed(system_prompt: str) -> None:
    llm_controller.session.set_system_prompt("custom", system_prompt)


def vote(data: gr.LikeData):
    action = "upvoted" if data.liked else "downvoted"
    print(f"You {action} this response: [{data.value}]")  # noqa: T201
    # gr.Info(f"You {action} this response: [{data.value}]")


def add_user_message(user_prompt: str, chat_history):
    chat_message = (user_prompt, None)
    return (
        "",
        chat_history + [chat_message],
        gr.Button(visible=False),
        gr.Button(visible=True),
    )


def _handle_response(response_type, response_text, chat_history):
    if response_type == "error":
        raise gr.Error(f"{response_type.upper()}: {response_text}")
    elif response_type == "warning":
        gr.Warning(f"{response_type.upper()}: {response_text}")
    elif response_type == "done":
        # Do Nothing if it is `done`
        pass
    else:
        # Add it to the chat_history
        chat_history[-1][1] = response_text


async def submit_query(state: gr.State, chat_history):
    # Pop out unsupported kwargs
    params = llm_controller.session.get_model_params()
    kwargs = {k: v for k, v in state["params"].items() if k in params}
    user_query = chat_history[-1][0]
    verbose = state.get("verbose", False)
    if state["stream_mode"]:
        stream = llm_controller.session.run_stream(user_query, verbose=verbose, **kwargs)
        async for response_type, response_text in stream:  # type: ignore
            _handle_response(response_type, response_text, chat_history)
            yield chat_history, gr.Button(visible=False), gr.Button(visible=True)
        # Last yield to restore the submit button
        yield chat_history, gr.Button(visible=True), gr.Button(visible=False)
    else:
        batch_gen = llm_controller.session.run_batch(user_query, verbose=verbose, **kwargs)
        async for response_type, response_text in batch_gen:
            if response_type != "done":
                response = response_text.strip()
                _handle_response(response_type, response, chat_history)
        yield chat_history, gr.Button(visible=True), gr.Button(visible=False)


def stop_btn_clicked():
    # TODO: Need to check if we can/need to actually stop the query
    return gr.Button(visible=True), gr.Button(visible=False)


def clear_btn_clicked():
    llm_controller.clear_history()
    return []


# ===========================================================================================
# Main Logic. Setup Gradio UI
# ===========================================================================================

llm_controller = LLMController()
llm_session = llm_controller.change_model()


def setup_gradio(verbose=False):
    logger.info("Setting up Gradio...")
    # Gradio Interface
    # with gr.Blocks(title=title, theme=gr.themes.Base(), css=css) as demo:
    with gr.Blocks(title=title) as demo:
        state = gr.State([])
        gr.Markdown(SHARED_UI_WARNING)
        gr.Markdown(TITLE_MARKDOWN)
        with gr.Row():
            model_list = llm_controller.get_model_list()
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    model_list,
                    value=model_list[0] if model_list else None,
                    label="Model",
                    info=f"Choose from one of the supported models: {len(model_list)} found!",
                    visible=False,
                )
            with gr.Column(scale=8):
                with gr.Row():
                    stream_mode = gr.Radio(
                        [("Streaming", True), ("Batched", False)],
                        value=True,
                        label="Mode",
                        scale=2,
                        show_label=True,
                        inline=True,
                        visible=True,
                    )
                    system_prompt = gr.Textbox(
                        value=llm_controller.session.system_prompt,
                        label="System prompt (Optional)",
                        placeholder="Enter the System prompt and place enter",
                        scale=8,
                    )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    max_tokens = gr.Slider(elem_id="max_tokens", visible=False)
                    temperature = gr.Slider(elem_id="temperature", visible=False)
                    top_p = gr.Slider(elem_id="top_p", visible=False)
                    top_k = gr.Slider(elem_id="top_k", visible=False)
                    length_penalty = gr.Slider(elem_id="length_penalty", visible=False)
                    repeat_penalty = gr.Slider(elem_id="repeat_penalty", visible=False)
                    num_sequences = gr.Slider(elem_id="num_sequences", visible=False)

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    avatar_images=("./images/user.png", "./images/bot.png"),
                    bubble_full_width=False,
                    render_markdown=True,
                    line_breaks=True,
                    show_copy_button=True,
                    show_label=False
                    # layout="panel",  # or 'bubble' # [Deprecated]
                )
                with gr.Row(visible=True) as submit_row:  # noqa: F841
                    user_prompt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press ENTER",
                        scale=8,
                        visible=True,
                    ).style(container=False)
                    submit_btn = gr.Button(value="Submit", scale=2, visible=True, variant="primary")
                    stop_btn = gr.Button(value="Stop", scale=2, visible=False, variant="stop")
                    clear_btn = gr.ClearButton(
                        [chatbot],
                        value="🗑️ Clear history",
                        scale=2,
                        variant="secondary",
                        interactive=True,
                    )

        gr.Examples(examples=examples, examples_per_page=20, inputs=user_prompt)

        parameters = [
            max_tokens,
            temperature,
            top_p,
            top_k,
            length_penalty,
            repeat_penalty,
            num_sequences,
        ]

        # Event Handlers
        model_dropdown.change(
            close_parameters,
            inputs=[],
            outputs=[parameter_row],
        ).then(
            lambda x, y, z, *params: model_changed(x, y, z, parameters),
            inputs=[state, model_dropdown, chatbot, *parameters],
            outputs=[state, model_dropdown, chatbot, parameter_row, *parameters],
        )
        stream_mode.change(
            mode_changed,
            inputs=[state, stream_mode],
            outputs=[state],
        )
        system_prompt.change(
            system_prompt_changed,
            inputs=[system_prompt],
            outputs=[],
        )

        # Parameter Event handlers. Need to separately done!
        max_tokens.change(
            fn=lambda state, y: parameter_changed(state, max_tokens, y),
            inputs=[state, max_tokens],
            outputs=[state],
        )
        temperature.change(
            fn=lambda state, y: parameter_changed(state, temperature, y),
            inputs=[state, temperature],
            outputs=[state],
        )
        top_p.change(
            fn=lambda state, y: parameter_changed(state, top_p, y),
            inputs=[state, top_p],
            outputs=[state],
        )
        top_k.change(
            fn=lambda state, y: parameter_changed(state, top_k, y),
            inputs=[state, top_k],
            outputs=[state],
        )
        length_penalty.change(
            fn=lambda state, y: parameter_changed(state, length_penalty, y),
            inputs=[state, length_penalty],
            outputs=[state],
        )
        repeat_penalty.change(
            fn=lambda state, y: parameter_changed(state, repeat_penalty, y),
            inputs=[state, repeat_penalty],
            outputs=[state],
        )
        num_sequences.change(
            fn=lambda state, y: parameter_changed(state, num_sequences, y),
            inputs=[state, num_sequences],
            outputs=[state],
        )

        chatbot.like(vote, None, None)
        user_prompt.submit(
            add_user_message,
            [user_prompt, chatbot],
            [user_prompt, chatbot, submit_btn, stop_btn],
            queue=False,
        ).then(
            submit_query,
            inputs=[state, chatbot],
            outputs=[chatbot, submit_btn, stop_btn],
        )
        submit_btn.click(
            add_user_message,
            [user_prompt, chatbot],
            [user_prompt, chatbot, submit_btn, stop_btn],
            queue=False,
        ).then(
            submit_query,
            inputs=[state, chatbot],
            outputs=[chatbot, submit_btn, stop_btn],
        )
        stop_btn.click(stop_btn_clicked, [], outputs=[submit_btn, stop_btn])
        clear_btn.click(clear_btn_clicked, [], outputs=[chatbot])

        demo.load(
            lambda x, *params: load_demo(x, parameters, verbose=verbose),
            [model_dropdown, *parameters],
            [state, model_dropdown, parameter_row, *parameters],
        )

    return demo
