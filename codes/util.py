import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Any
from math import ceil
import argparse
from tqdm import tqdm
import pandas as pd
import sys, os

def parse_assistantfinal(text):
    """
    Return the substring after the last occurrence of 'assistantfinal'.
    Works with str, list/tuple, and pandas.Series. Preserves NaN and other types.
    If 'assistantfinal' is not present, returns the original input unchanged.
    """
    if isinstance(text, pd.Series):
        return text.astype(object).apply(parse_assistantfinal)
    if isinstance(text, list):
        return [parse_assistantfinal(t) for t in text]
    if isinstance(text, tuple):
        return tuple(parse_assistantfinal(t) for t in text)
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    if isinstance(text, str):
        if "assistantfinal" in text:
            return text.split("assistantfinal")[-1].strip()
        return text
    return text

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
    SYSTEM_PROMPT = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )



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
