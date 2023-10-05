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

system_message = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
css = """.toast-wrap { display: none !important } """

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
    state = {"active_tab": "streaming_tab", "model": model_name, "params": values}
    logger.info(f"Loaded Demo: state = {state}")
    return (
        state,
        model_name,
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
    logger.debug(f"Locals = {locals()}")
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


def vote(data: gr.LikeData):
    action = "upvoted" if data.liked else "downvoted"
    print(f"You {action} this response: [{data.value}]")
    # gr.Info(f"You {action} this response: [{data.value}]")


def tab_changed(state: gr.State, active_tab: str) -> gr.State:
    logger.debug(f"Locals = {locals()}")
    state.update({"active_tab": active_tab})
    return state


def add_user_message(state: gr.State, user_input: str, stream_history, batch_history):
    logger.debug(f"Locals [add_user_message] = {locals()}")
    chat_message = (user_input, None)
    if state["active_tab"] == "streaming_tab":
        return ("", stream_history + [chat_message], batch_history)
    else:
        return ("", stream_history, batch_history + [chat_message])


async def submit_query(state: gr.State, stream_history, batch_history, system_prompt: str):
    logger.info(f"Locals [submit_query] = {locals()} / State = {state}")
    kwargs = copy.copy(state["params"])
    kwargs.pop("top_k", None)
    kwargs.pop("length_penalty", None)
    if state["active_tab"] == "streaming_tab":
        query = stream_history[-1][0]
        logger.info(f"Running Streaming Query: {query} / {kwargs}")
        stream = llm_controller.run_stream(query, system_prompt=system_prompt, **kwargs)
        async for response_text in stream:
            stream_history[-1][1] = response_text
            yield stream_history, batch_history
    else:
        query = batch_history[-1][0]
        logger.info(f"Running Batched Query: {query}")
        response_text = await llm_controller.run_query(
            query, system_prompt=system_prompt, **kwargs
        )
        batch_history[-1][1] = response_text
        yield stream_history, batch_history


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
                )
            with gr.Column(scale=8):
                system_prompt = gr.Textbox("", label="System prompt (Optional)")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("Parameters", open=True, visible=True) as parameter_row:
                    max_tokens = gr.Slider(
                        minimum=0,
                        maximum=5000,  # data['params']['context_length']
                        value=500,  # data["params"]["max_tokens"],
                        step=50,
                        interactive=True,
                        label="Max output tokens",
                        info="The maximum numbers of new tokens",
                        elem_id="max_tokens",
                    )
                    temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1,
                        value=0.7,  # data["params"]["temperature"],
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                        info="Higher values produce more diverse outputs",
                        elem_id="temperature",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,  # data["params"]["top_k"],
                        step=1,
                        interactive=True,
                        label="Top k",
                        elem_id="top_k",
                    )
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=1,  # data["params"]["top_p"],
                        step=0.1,
                        interactive=True,
                        label="Top p",
                        info="Alternative to temperature sampling, called nucleus sampling",
                        elem_id="top_p",
                    )
                    length_penalty = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,  # data["params"]["length_penalty"],
                        step=0.1,
                        interactive=True,
                        label="length_penalty",
                        elem_id="length_penalty",
                    )

            with gr.Column(scale=8):
                with gr.Tabs() as mode_tabs:
                    with gr.Tab("Streaming", id="streaming_tab") as streaming_tab:
                        chatbot_stream = gr.Chatbot(
                            avatar_images=("./images/user.png", "./images/bot.png"),
                            bubble_full_width=False,
                            # layout="panel",  # or 'bubble' # [Deprecated]
                        )
                    with gr.Tab("Batched", id="batched_tab") as batched_tab:
                        chatbot_batch = gr.Chatbot(
                            avatar_images=("./images/user.png", "./images/bot.png"),
                            bubble_full_width=False,
                            # layout="bubble",  # or 'panel' # [Deprecated]
                        )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            visible=True,
                        ).style(container=False)
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=True)
                with gr.Row(visible=True) as button_row:
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=True)
                    clear_btn = gr.ClearButton(
                        [chatbot_stream, chatbot_batch], value="🗑️ Clear history", interactive=True
                    )

        gr.Examples(examples=examples, inputs=textbox)
        # url_params = gr.JSON(visible=False)  # TODO: Why??
        # btn_list = [flag_btn, clear_btn]

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

        streaming_tab.select(
            lambda state: tab_changed(state, "streaming_tab"), inputs=[state], outputs=[state]
        )
        batched_tab.select(
            lambda state: tab_changed(state, "batched_tab"), inputs=[state], outputs=[state]
        )
        chatbot_stream.like(vote, None, None)
        chatbot_batch.like(vote, None, None)

        textbox.submit(
            add_user_message,
            [state, textbox, chatbot_stream, chatbot_batch],
            [textbox, chatbot_stream, chatbot_batch],
            queue=False,
        ).then(
            submit_query,
            inputs=[state, chatbot_stream, chatbot_batch, system_prompt],
            outputs=[chatbot_stream, chatbot_batch],
        )
        submit_btn.click(
            add_user_message,
            [state, textbox, chatbot_stream, chatbot_batch],
            [textbox, chatbot_stream, chatbot_batch],
            queue=False,
        ).then(
            submit_query,
            inputs=[state, chatbot_stream, chatbot_batch, system_prompt],
            outputs=[chatbot_stream, chatbot_batch],
        )

        # flag_btn.click(flag_last_response, [state], [textbox, upvote_btn, downvote_btn, flag_btn])
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
                max_tokens,
                temperature,
                top_k,
                top_p,
                length_penalty,
            ],
            # _js=get_window_url_params,
        )

    return demo
