import argparse
import os

import torch
from dotenv import dotenv_values, load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


def hf(model_name, prompt):
    print(f"Model Name = {model_name}")
    print(f"Prompt = {prompt}")
    llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    kwargs = {
        "do_sample": True,
        "max_new_tokens": 128,
        "temperature": 1.0,
        "top_k": 3,
        "top_p": 0.9,
        "length_penalty": 1,
        "num_beams": 4,
        "num_return_sequences": 4,
    }
    print(
        f"Validated kwargs = {kwargs}, Input tokens = {input_tokens.shape}, [{len(prompt)} chars"
    )
    hf_response = llm.generate(input_tokens, **kwargs)
    print(f"Response shape = {hf_response.shape}")
    out_tokens = torch.numel(hf_response)
    print(f"Out Tokens = {out_tokens}")
    sequences = []
    for seq in hf_response:
        seq_texts = tokenizer.batch_decode(seq, skip_special_tokens=True)
        sequences.append(seq_texts)
    for i, seq in enumerate(sequences):
        text_seq = "".join(seq)
        print(f"{i+1}: {text_seq}")
    print()  # newline
    return sequences


prompt = """
System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
User: Can you explain to me briefly what is Python programming language?
"""
# model = "microsoft/phi-1_5"
# prompt = "My name is Lewis and I like to "
# model = "roneneldan/TinyStories-33M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hfcheck", description="HF Check/Download Models")
    parser.add_argument("-m", "--model", type=str, default="gpt2")
    parser.add_argument("-v", "--verbose", action="store_true", help="using verbose mode")
    args = parser.parse_args()
    if args.verbose:
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


if __name__ == "__main__":
    set_env(debug=True)
    args = parse_args()
    hf(args.model, prompt)
