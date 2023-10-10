import torch
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


prompt = "System: You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. User: Can you explain to me briefly what is Python programming language?"
model = "gpt2"
# model = "microsoft/phi-1_5"
# prompt = "My name is Lewis and I like to "
# model = "roneneldan/TinyStories-33M"
hf(model, prompt)
