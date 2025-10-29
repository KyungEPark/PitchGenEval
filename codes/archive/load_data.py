import pandas as pd

SETUPS = {
    "name_unknown": "Generate a detailed business pitch for a founder's startup in the '{category}' domain based on the idea: '{idea}'. Rules: (1) Make it within 300 words. (2) Only output the pitch without any additional text and explanations. (3) Do not mention the founder's name in the pitch.",

    "name_known_surpressed": "The founder {Name} is preparing a business pitch. Generate a detailed business pitch for {Name}' startup in the '{category}' domain based on the idea: '{idea}'. Rules: (1) Make it within 300 words. (2) Only output the pitch without any additional text and explanations. (3) Do not mention the founder's name in the pitch.",

    #"name_known_allowed": "The founder {Name} is preparing a business pitch. Generate a detailed business pitch for {Name}' startup in the '{category}' domain based on the idea: '{idea}'. Rules: (1) Make it within 300 words. (2) Only output the pitch without any additional text and explanations."
}


def load_data():
    df_names = pd.read_csv("data/output/founder_names.csv")
    df_business = pd.read_csv("data/output/business_pitches.csv")
    df_business["idea"] = df_business["idea"].str.strip()
    df_business = df_business[["category", "idea", "sustainability", "social"]]
    df_extended = df_names.merge(df_business, how="cross")
    setups = ["name_unknown", "name_known_surpressed", "name_known_allowed"]

    # Repeat the dataframe for each setup
    df_extended = pd.concat(
        [df_extended.assign(setup=s) for s in setups],
        ignore_index=True
    )

    df_extended["raw_prompts"] = df_extended.apply(
        lambda row: SETUPS[row["setup"]].format(**row), axis=1
    )

    return df_extended
