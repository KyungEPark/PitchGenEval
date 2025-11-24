#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from codes.gen_eval import add_eval
from codes.util import *

SYSTEM_PROMPT = "You are a helpful assistant."

import os

def load_model_and_tokenizer(model_name: str):
    local_dir = f"/p/project1/westai0091/Models/{model_name}"
    model_path = local_dir

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


'''
def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # let HF handle device placement
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer
'''

def extract_generated_part(tokenizer, model_inputs, gen):
    outputs = []
    for g, inp in zip(gen, model_inputs["input_ids"]):
        # decode full input prompt
        prompt_text = tokenizer.decode(
            inp,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
 
        # decode full generated sequence
        text = tokenizer.decode(
            g,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
 
        # keep only continuation
        continuation = text[len(prompt_text):].strip()
        outputs.append(continuation)
 
    return outputs

def build_chat_text(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    user_prompt = str(user_prompt)
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

def strip_prompt(prompt_text: str, full_output: str, tokenizer):
    # Tokenize prompt and output
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    output_ids = tokenizer(full_output, add_special_tokens=False).input_ids

    # Edge case: output shorter than prompt
    if len(output_ids) <= len(prompt_ids):
        return full_output.strip()

    # Keep only the newly generated part
    gen_ids = output_ids[len(prompt_ids):]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

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
    for bi in tqdm(range(num_batches)):
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
        generated_texts = extract_generated_part(tokenizer, model_inputs, gen)
        outputs += generated_texts
    return outputs, chat_texts


def main():
    parser = argparse.ArgumentParser(description="Batch generation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    project_root = os.path.dirname(os.getcwd())
    default_output_folder = os.path.join(project_root, "venturebias", "data", "output", "eval")
    parser.add_argument("--output_folder", type=str, default=default_output_folder,
                        help="File containing prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the Data
    df = add_eval(args.model_name)
    prompts = df["eval_prompt"].tolist()

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Inference
    full_outputs, chat_texts = generate_batch(
        model,
        tokenizer,
        prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample
    )
    full_outputs = parse_assistantfinal(full_outputs)
    df["eval"] = full_outputs
    os.makedirs(args.output_folder, exist_ok=True)
    df.to_csv(f"{args.output_folder}/{args.model_name}_eval_with_evalname.csv", index=False) # PLEASE DELETE _TEST AFTER TESTING / _Evalname
    print(f"Generation completed. Results saved to {args.output_folder}/{args.model_name.split('/')[-1]}_eval.csv")


if __name__ == "__main__":
    main()
