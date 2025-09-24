# OpenAI calling
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI


def call_openai(prompt, text, model="gpt-3.5-turbo",  max_tokens=1000):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a start-up expert."},
                  {"role": "user", "content": prompt}
                  ],
        max_tokens=max_tokens,
        temperature = 0.7,
    )
    return response.choices[0].message['content']

def get_evaluation(prompt, chunk, model_name, hf_token=None, max_new_tokens=40, device=None):
    """
    Evaluate a pitch (chunk) using a Hugging Face causal/chat model.
    Supports locked models via hf_token (passed to from_pretrained as use_auth_token).
    Returns a tuple: (decoded_response_str, parsed_scores_dict).
    parsed_scores_dict has keys: novelty, viability, environmental, financial, overall
    Values are ints 1-5 or None if parsing failed.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import re

    def parse_scores(text: str):
        """
        Parse the model response text and extract the 5 numeric scores (1-5).
        Returns a dict with keys: novelty, viability, environmental, financial, overall.
        """
        text = text or ""
        keys_aliases = {
            "novelty": r"novelty",
            "viability": r"viability|strategic viability",
            "environmental": r"environmental(?: value)?|enviro",
            "financial": r"financial(?: value)?",
            "overall": r"overall"
        }
        scores = {}
        for key, alias in keys_aliases.items():
            # look for formats like "novelty: 4", "novelty - 4", "novelty 4/5", "novelty: 4/5", etc.
            pattern = rf"(?i){alias}\s*[:\-]?\s*([1-5])"
            m = re.search(pattern, text)
            if not m:
                # fallback: look for the key somewhere near a digit 1-5 (within 15 chars)
                pattern2 = rf"(?i){alias}.{0,15}?([1-5])"
                m = re.search(pattern2, text)
            scores[key] = int(m.group(1)) if m else None
        return scores

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model (supports locked models via use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)

    grading_instructions = (
        "Evaluate the following pitch one-pager on a scale from 1 to 5 for each criteria: "
        "1) solution's novelty (how different is it from existing solutions?), "
        "2) strategic viability (how likely is it to succeed and how scalable is it?), "
        "3) environmental value (how much does it benefit the planet?), "
        "4) financial value (what financial value can it create for businesses?), and "
        "5) overall quality (based on the four criteria above, what is the overall quality of the pitch?) "
        "Answer in the format of: novelty: <1-5>, viability: <1-5>, environmental: <1-5>, financial: <1-5>, overall: <1-5>."
    )

    # Build the messages/prompt the model will consume
    system_msg = {"role": "system", "content": "You are an assistant to venture capitalists."}
    user_msg1 = {"role": "user", "content": grading_instructions + "\n\n" + prompt}
    assistant_ack = {"role": "assistant", "content": "Ok"}
    user_msg2 = {"role": "user", "content": chunk}

    # If tokenizer supports chat template, use it (keeps instructions/roles)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [system_msg, user_msg1, assistant_ack, user_msg2]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_dict=True,
            tokenize=True,
        ).to(device)
        # Generate
        with torch.no_grad():
            pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id, do_sample=False)
            # Slice off the input tokens to decode only the generated part
            if "input_ids" in inputs:
                generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            else:
                generated_tokens = outputs[0]
            decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            parsed = parse_scores(decoded)
            return decoded, parsed

    # Fallback for non-chat tokenizers/models: concatenate into a single prompt
    full_prompt = grading_instructions + "\n\n" + prompt + "\n\n" + chunk
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(device)
    with torch.no_grad():
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id, do_sample=False)
        # slice input length to get only generated tokens
        gen_start = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][gen_start:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        parsed = parse_scores(decoded)
        return decoded, parsed


def append_scores_to_df(df, response_col, prefix="", keys=None):
    """
    Given a DataFrame df and a column name that contains the model response text,
    parse the five scores from each response and append them as new columns.
    - response_col: name of the column containing the model's decoded text (string).
    - prefix: optional string to prefix new column names (e.g. "eval_").
    - keys: optional list of keys/order to use. Default: ['novelty','viability','environmental','financial','overall'].
    Returns a new DataFrame with the added columns.
    """
    import pandas as pd
    import re

    keys = keys or ['novelty', 'viability', 'environmental', 'financial', 'overall']

    def _parse_scores_simple(text):
        text = text or ""
        aliases = {
            "novelty": r"novelty",
            "viability": r"viability|strategic viability",
            "environmental": r"environmental(?: value)?|enviro",
            "financial": r"financial(?: value)?",
            "overall": r"overall"
        }
        out = {}
        for k in keys:
            alias = aliases.get(k, k)
            m = re.search(rf"(?i){alias}\s*[:\-]?\s*([1-5])", text)
            if not m:
                m = re.search(rf"(?i){alias}.{0,15}?([1-5])", text)
            out[k] = int(m.group(1)) if m else None
        return out

    # parse each response into dicts, then normalize into columns
    parsed_series = df[response_col].fillna("").apply(_parse_scores_simple)
    parsed_df = pd.json_normalize(parsed_series)
    # ensure all keys exist
    for k in keys:
        if k not in parsed_df:
            parsed_df[k] = None
    parsed_df = parsed_df[keys].reset_index(drop=True)
    parsed_df.columns = [f"{prefix}{c}" for c in parsed_df.columns]
    result = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    return result



# Optional helper kept for convenience (uses similar loading logic)
def load_huggingface_model_locked(model_name, hf_token=None, device=None):
    """
    Load tokenizer and model from HF, supports locked repos via hf_token.
    Returns (tokenizer, model) with model moved to device.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token).to(device)
    return tokenizer, model

