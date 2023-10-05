import os

import gradio as gr
from chatllm.constants import (  # noqa: F401
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_MAX_TOKENS,
)

# Import this after setting the env.
from chatllm.llm_controller import LLMController

title = "ChatLLM: A Chatbot for Language Models"
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


def model_changed(model: str, dropdown: gr.Dropdown):
    try:
        print(f"Model Changed from {dropdown.value} to {model}")
        llm_controller.load_model(model)
    except Exception as e:
        print(f"Error loading model {model}: {e}")
        gr.Info(f"Unable to load model {model}... Using {dropdown.value}")


def vote(data: gr.LikeData):
    action = "upvoted" if data.liked else "downvoted"
    print(f"You {action} this response: [{data.value}]")
    # gr.Info(f"You {action} this response: [{data.value}]")


additional_inputs = [
    gr.Textbox("", label="Optional system prompt"),
]

# ===========================================================================================
# Main Logic. Setup Gradio UI
# ===========================================================================================

llm_controller = LLMController()
model_list = llm_controller.get_model_list()
llm_controller.load_model()


def setup_gradio(verbose=False):
    # Needs to be defined outside the block so that it is not rendered..
    # We only want the avatar_images to be rendered and the vote function to be called
    chatbot_batch = gr.Chatbot(
        avatar_images=("./images/user.png", "./images/bot.png"),
        bubble_full_width=False,
        # layout="bubble",  # or 'panel' # [Deprecated]
    )
    chatbot_stream = gr.Chatbot(
        avatar_images=("./images/user.png", "./images/bot.png"),
        bubble_full_width=False,
        # layout="panel",  # or 'bubble' # [Deprecated]
    )

    # Gradio Interface
    with gr.Blocks(title=title) as gr_app:
        gr.Markdown(f"<h1 style='text-align: center;'>{title}</h1>")
        with gr.Row():
            model_dropdown = gr.Dropdown(
                model_list,
                label="Model",
                value=llm_controller.get_default_model(),
                info=f"Choose from one of the supported models: {len(model_list)} found!",
            )
            model_dropdown.change(
                fn=lambda x: model_changed(x, model_dropdown), inputs=model_dropdown
            )
            temp_slider = gr.Slider(
                label="Temperature",
                value=DEFAULT_TEMPERATURE,
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                interactive=True,
                info="Higher values produce more diverse outputs",
            )
            max_tokens_slider = gr.Slider(
                label="Max new tokens",
                value=DEFAULT_MAX_TOKENS,
                minimum=0,
                maximum=MAX_MAX_TOKENS,
                step=50,
                interactive=True,
                info="The maximum numbers of new tokens",
            )

        with gr.Tab("Streaming"):
            chatbot_stream.like(vote, None, None)
            chat_interface_stream = gr.ChatInterface(
                llm_controller.run_stream,
                # title=title,
                # css=css,
                chatbot=chatbot_stream,
                additional_inputs=additional_inputs + [temp_slider, max_tokens_slider],
                examples=examples,
                # cache_examples=True,
            )
            chatbot_stream.like(vote, None, None)

        with gr.Tab("Batched"):
            chatbot_batch.like(vote, None, None)
            chat_interface_batch = gr.ChatInterface(
                llm_controller.run_query,
                # title=title,
                # css=css,
                chatbot=chatbot_batch,
                additional_inputs=additional_inputs + [temp_slider, max_tokens_slider],
                examples=examples,
                # cache_examples=True,
            )
            chatbot_batch.like(vote, None, None)

    return gr_app
