# -*- coding: utf-8 -*-
"""
generate.py

Use the fine-tuned GPT-2 small model for text generation tests.
"""

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    pipeline
)

def load_model(model_dir: str):
    """
    Load the tokenizer and model, and move the model to GPU if available.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_once(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int = 50,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
):
    """
    Generate a continuation for a single prompt using model.generate.
    Returns the generated text as a string.
    """
    inputs = tokenizer(prompt, return_tensors="pt",truncation=True,max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_batch(
    tokenizer,
    model,
    device,
    prompt: str,
    max_length: int = 100,
    num_return_sequences: int = 3,
    top_p: float = 0.9,
):
    """
    Generate multiple sequences for a single prompt using the HuggingFace pipeline.
    Returns a list of dictionaries with key 'generated_text'.
    """
    device_id = 0 if torch.cuda.is_available() else -1
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )
    return text_generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True
    )


def main():
    model_directory = "./logs/final_model"
    tokenizer, model, device = load_model(model_directory)

    # Single-sequence generation example
    print("=== Single generation (model.generate) ===")
    prompt_text = "Once upon a time"
    result = generate_once(
        tokenizer,
        model,
        device,
        prompt_text,
        max_new_tokens=100,
        top_k=30,
        top_p=0.8,
        temperature=1.0
    )
    print(result)

    # Multi-sequence generation example
    print("\n=== Multiple generations (pipeline) ===")
    prompt2 = "In a distant future, humanity has"
    choices = generate_batch(
        tokenizer,
        model,
        device,
        prompt2,
        max_length=100,
        num_return_sequences=3,
        top_p=0.8
    )
    for idx, item in enumerate(choices, 1):
        print(f"\n--- Option {idx} ---")
        print(item["generated_text"])


if __name__ == "__main__":
    main()
