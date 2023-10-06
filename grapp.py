"""Setup the Gradio App"""
import copy
import logging

import gradio as gr
from altair import param

# Import this after setting the env.
from chatllm.llm_controller import LLMController

logger = logging.getLogger(__name__)

# ===========================================================================================
# Constants
# ===========================================================================================
SHARED_UI_WARNING = f"""
"""

title = "ChatLLM: A Chatbot for Language Models"
TITLE_MARKDOWN = f"""
<h1 style='text-align: center;'>{title}</h1>
"""

simple_system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe."
long_system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
"""

examples = [
    ["Hello there! How are you doing?"],
    ["Can you explain to me briefly what is Python programming language?"],
    [
        "please write a python program to find the first n numbers of the fibonacci series, where n is the input variable"
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
        "In bash, how do i list all the text files in the current directory that have been modified in the last month"
    ],
    ["Name the planets in the solar system?"],
]


def get_param_values(params, param_name: str, default_value: int | float):
    if param_name in params and isinstance(params[param_name], dict):
        value = params[param_name].get("default", default_value)
        kwargs = {
            "value": value,
            "minimum": params[param_name]["min"],
            "maximum": params[param_name]["max"],
            "step": params[param_name]["step"],
            "visible": param_name in params.keys(),
        }
    else:
        value = params.get(param_name, default_value)
        kwargs = {
            "value": value,
            "visible": param_name in params.keys(),
        }
    return kwargs


def load_demo(model_name, max_tokens, temperature, top_k, top_p, length_penalty):
    """Load the Demo"""
    # Get all the parameters from the local arguments
    local_args = {k: v for k, v in locals().items() if k != "model_name"}
    llm_controller.load_model(model_name)
    params = llm_controller.get_model_params(model_name)
    param_values = {k: get_param_values(params, k, local_args[k]) for k in local_args.keys()}
    values = {k: v["value"] for k, v in param_values.items()}
    state = {"stream_mode": True, "model": model_name, "params": values}
    logger.info(f"Loaded Demo: state = {state}")
    return (
        state,
        gr.Dropdown(model_name, visible=True),  # Model Dropdown - Make visible
        gr.Accordion(open=True, visible=True),  # parameter_row - Open and visible
        gr.Slider(**param_values["max_tokens"]),
        gr.Slider(**param_values["temperature"]),
        gr.Slider(**param_values["top_k"]),
        gr.Slider(**param_values["top_p"]),
        gr.Slider(**param_values["length_penalty"]),
    )


# Event Handlers Implemented
def model_changed(
    state: gr.State, model_name: str, max_tokens, temperature, top_k, top_p, length_penalty
):
    """Handle Model Change"""
    local_args = {k: v for k, v in locals().items() if k != "model_name"}
    param_values = state["params"]
    if model_name != state["model"]:
        """Load Model only if the model has changed"""
        try:
            logger.info(f"Changing model from {state['model']} to {model_name}")
            llm_controller.load_model(model_name)
            params = llm_controller.get_model_params(model_name)
            param_values = {
                k: get_param_values(params, k, local_args[k]) for k in local_args.keys()
            }
            values = {k: v["value"] for k, v in param_values.items()}
            state["params"] = values
            state["model"] = model_name
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            gr.Info(f"Unable to load model {model_name}... Using {state['model']}")
    else:
        logger.info(f"Model {model_name} has not changed")
    return (
        state,
        state["model"],
        gr.Slider(**param_values["max_tokens"]),
        gr.Slider(**param_values["temperature"]),
        gr.Slider(**param_values["top_k"]),
        gr.Slider(**param_values["top_p"]),
        gr.Slider(**param_values["length_penalty"]),
    )


def parameter_changed(state: gr.State, parameter: gr.Slider, value: int | float):
    logger.debug(f"Parameter: {parameter.elem_id} // Value: {value} / {parameter.value}")
    state["params"][parameter.elem_id] = value
    return state


def mode_changed(state: gr.State, active_mode: str):
    state.update({"stream_mode": active_mode})
    return state


def vote(data: gr.LikeData):
    action = "upvoted" if data.liked else "downvoted"
    print(f"You {action} this response: [{data.value}]")
    # gr.Info(f"You {action} this response: [{data.value}]")


def add_user_message(user_prompt: str, chat_history):
    chat_message = (user_prompt, None)
    return ("", chat_history + [chat_message], gr.Button(visible=False), gr.Button(visible=True))


async def submit_query(state: gr.State, chat_history, system_prompt: str):
    # Pop out unsupported kwargs
    params = llm_controller.get_model_params(state["model"])
    kwargs = {k: v for k, v in state["params"].items() if k in params}
    query = chat_history[-1][0]
    mode = "stream" if state["stream_mode"] else "batch"
    logger.info(f"Running {mode} Query: {query} / {kwargs}")
    if state["stream_mode"]:
        stream = llm_controller.run_stream(query, system_prompt=system_prompt, **kwargs)
        async for response_text in stream:
            chat_history[-1][1] = response_text
            yield chat_history, gr.Button(visible=False), gr.Button(visible=True)
        # Last yield to restore the submit button
        yield chat_history, gr.Button(visible=True), gr.Button(visible=False)
    else:
        response_text = await llm_controller.run_query(
            query, system_prompt=system_prompt, **kwargs
        )
        chat_history[-1][1] = response_text
        yield chat_history, gr.Button(visible=True), gr.Button(visible=False)


# ===========================================================================================
# Main Logic. Setup Gradio UI
# ===========================================================================================

llm_controller = LLMController()


def setup_gradio(verbose=False):
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
                    value=llm_controller.get_default_model(),
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
                        value=simple_system_prompt,  # or long_system_prompt
                        label="System prompt (Optional)",
                        placeholder="Enter the System prompt and place enter",
                        scale=8,
                    )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                    max_tokens = gr.Slider(
                        minimum=0,
                        maximum=5000,
                        value=500,
                        step=50,
                        interactive=True,
                        label="Max output tokens",
                        info="The maximum numbers of new tokens",
                        elem_id="max_tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                        info="Higher values produce more diverse outputs",
                        elem_id="temperature",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        interactive=True,
                        label="Top k",
                        elem_id="top_k",
                    )
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=1,
                        step=0.1,
                        interactive=True,
                        label="Top p",
                        info="Alternative to temperature sampling, called nucleus sampling",
                        elem_id="top_p",
                    )
                    length_penalty = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=0.1,
                        interactive=True,
                        label="length_penalty",
                        elem_id="length_penalty",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    avatar_images=("./images/user.png", "./images/bot.png"),
                    bubble_full_width=False,
                    # layout="panel",  # or 'bubble' # [Deprecated]
                )
                with gr.Row(visible=True) as submit_row:
                    user_prompt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press ENTER",
                        scale=8,
                        visible=True,
                    ).style(container=False)
                    submit_btn = gr.Button(
                        value="Submit", scale=2, visible=True, variant="primary"
                    )
                    stop_btn = gr.Button(value="Stop", scale=2, visible=False, variant="stop")
                    gr.ClearButton(
                        [chatbot],
                        value="🗑️ Clear history",
                        scale=2,
                        variant="secondary",
                        interactive=True,
                    )

        gr.Examples(examples=examples, inputs=user_prompt)

        # Event Handlers
        model_dropdown.change(
            model_changed,
            inputs=[state, model_dropdown, max_tokens, temperature, top_k, top_p, length_penalty],
            outputs=[state, model_dropdown, max_tokens, temperature, top_k, top_p, length_penalty],
        )
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
        top_k.change(
            fn=lambda state, y: parameter_changed(state, top_k, y),
            inputs=[state, top_k],
            outputs=[state],
        )
        top_p.change(
            fn=lambda state, y: parameter_changed(state, top_p, y),
            inputs=[state, top_p],
            outputs=[state],
        )
        length_penalty.change(
            fn=lambda state, y: parameter_changed(state, length_penalty, y),
            inputs=[state, length_penalty],
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
            inputs=[state, chatbot, system_prompt],
            outputs=[chatbot, submit_btn, stop_btn],
        )
        submit_btn.click(
            add_user_message,
            [user_prompt, chatbot],
            [user_prompt, chatbot, submit_btn, stop_btn],
            queue=False,
        ).then(
            submit_query,
            inputs=[state, chatbot, system_prompt],
            outputs=[chatbot, submit_btn, stop_btn],
        )
        stream_mode.change(
            mode_changed,
            inputs=[state, stream_mode],
            outputs=[state],
        )

        demo.load(
            load_demo,
            [
                model_dropdown,
                max_tokens,
                temperature,
                top_k,
                top_p,
                length_penalty,
            ],
            [
                state,
                model_dropdown,
                parameter_row,
                max_tokens,
                temperature,
                top_k,
                top_p,
                length_penalty,
            ],
        )

    return demo
