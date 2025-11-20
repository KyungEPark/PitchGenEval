import os
import pandas as pd

folder = "/Users/duc/Desktop/Projects/Ongoing/PitchGenEval/plots/output"  # change this

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        # prepend the text
        df["response_with_name"] = df.apply(
            lambda row: f"Name of Founder: {row['Name']}\n\n{row['response']}",
            axis=1,
        )

        # save back to the same file
        df.to_csv(path, index=False)
        print(f"Updated {file}")
