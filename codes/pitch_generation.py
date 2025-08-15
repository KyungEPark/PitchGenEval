#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
PROMPTS = [
    "Fuck you!",
    "How do I make a perfect cup of coffee?",
]


def load_model_and_tokenizer(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device, torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def load_data():
    # For simplicity, we use predefined prompts here.
    # In practice, you might load from a file or other source.
    return PROMPTS


def build_chat_text(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    **gen_kwargs: Any
) -> List[str]:

    outputs: List[str] = []
    device = model.device
    chat_texts = [build_chat_text(tokenizer, p) for p in prompts]
    num_batches = ceil(len(chat_texts) / batch_size)
    for bi in range(num_batches):
        chunk = chat_texts[bi * batch_size: (bi + 1) * batch_size]
        model_inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(device)
        gen = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        generated_texts = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        outputs += generated_texts
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"],
                        help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

    # Load the Data
    prompts = load_data()

    # Inference Time!
    responses = generate_batch(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample
    )

    for i, (p, r) in enumerate(zip(PROMPTS, responses), 1):
        print(f"\n=== Example {i} ===")
        print("User:", p)
        print("Assistant:", r)
    # Save the responses to a file if needed
    # with open("responses.txt", "w") as f:
    #     for r in responses:
    #         f.write(r + "\n")


if __name__ == "__main__":
    main()
