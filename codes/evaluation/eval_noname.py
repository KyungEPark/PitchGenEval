import pandas as pd
from util import *
import huggingface
import argparse
import os
import pandas as pd

def main(model_name, savefile):
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        token=hf_token
    )

    # Load pitches
    pitches = pd.read_csv('data/output/business_pitches.csv')
    names = pd.read_csv('data/input/founder_names.csv')

    pitches_with_names = []

    for pitch in pitches['pitch']:
        for index, row in names.iterrows():
            pitch_w_name = "Founder name:  " + row['Name'] + "\n" + pitch
            country = row['Country']
            gender = row['Gender']
            pitches_with_names.append(pitch_w_name)


    # Efficiently process each dataframe
    for df, text_col, label_col, prompt in prompts:
        print(f"Processing {label_col} for {len(df)} rows")
        df[label_col] = [
            sliding_window_labeling(prompt, text, model, tokenizer)[0]
            for text in df[text_col]
        ]
        # Save each dataframe separately
        for df, _, label_col, _ in prompts:
            save_path = os.path.join('data/output', f"{label_col}_{os.path.basename(savefile)}")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_parquet(save_path, index=False)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model to be used")
    parser.add_argument("--savefile", type=str, required=True,
                        help="Save results")
    args = parser.parse_args()

    main(args.model, args.savefile)