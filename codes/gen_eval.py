import pandas as pd

PROMPT = ("Evaluate the following pitch one-pager on a scale from 1 to 5 for each criteria: " +
         "1) solution's novelty (how different is it from existing solutions?), " +
         "2) strategic viability (how likely is it to succeed and how scalable is it?), " +
         "3) environmental value (how much does it benefit the planet?), " +
         "4) financial value (what financial value can it create for businesses?), and " +
         "5) overall quality (based on the four criteria above, what is the overall quality of the pitch?) " +
         "Give only the scores without any explanation in the following format - novelty: <1-5>, viability: <1-5>, environmental: <1-5>, financial: <1-5>, overall: <1-5>.")

def add_eval(MODEL_NAME: str):
    df_pitches = pd.read_csv(f"data/output/pitches/{MODEL_NAME}_with_evalname.csv") # Delete _with_evalname
    df_pitches["eval_prompt"] = PROMPT + "\n\nPitch:\n" + df_pitches["response"]

    # !!!PLEASE COMMENT OUT AFTER TEST!!!
    #df_pitches = df_pitches.sample(n=10, random_state=42).reset_index(drop=True)

    return df_pitches
