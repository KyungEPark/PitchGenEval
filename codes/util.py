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


def get_evaluation(prompt, chunk, model, tokenizer):
    messages = [
        {"role": "user", "content": prompt + "Antworte nur mit Ja oder Nein."},
        {"role": "assistant", "content": "Ok"},
        {"role": "user", "content": chunk}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=40)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

def sliding_window_labeling(prompt, full_text, model, tokenizer, chunk_size=20000, stride=512):
    chunks = chunk_text(full_text, tokenizer, chunk_size, stride)
    predictions = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        prediction = get_chunk_prediction(prompt, chunk, model, tokenizer)
        predictions.append(prediction)

    # Prioritize "Ja" if present
    normalized_preds = [p.strip().lower() for p in predictions]
    if any("ja" in p for p in normalized_preds):
        return 1, predictions

    # Otherwise return most common label
    counter = Counter(predictions)
    majority_label, _ = counter.most_common(1)[0]
    return majority_label, predictions
