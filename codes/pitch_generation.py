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
from codes.gen_prompt import load_data
from codes.util import *


@torch.inference_mode()

def main():
    parser = argparse.ArgumentParser(description="Batch generation using Qwen model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    project_root = os.getcwd()
    default_output_folder = os.path.join(project_root, "data", "output", "pitches")
    parser.add_argument("--output_folder", type=str, default=default_output_folder,
                        help="File containing prompts for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true", help="Whether to sample instead of greedy decoding")

    args = parser.parse_args()

    # Load the Data
    df = load_data()
    prompts = df["raw_prompts"].tolist()

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
    df["response"] = full_outputs
    os.makedirs(args.output_folder, exist_ok=True)
    df.to_csv(f"{args.output_folder}/{args.model_name.split('/')[-1]}_test.csv", index=False)
    print(f"Generation completed. Results saved to {args.output_folder}/{args.model_name.split('/')[-1]}.csv")


if __name__ == "__main__":
    main()
