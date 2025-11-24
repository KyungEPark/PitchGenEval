import os
import pandas as pd

folder = "/p/project1/westai0091/venturebias/data/output/pitches"  # change this

for file in os.listdir(folder):
    if file.endswith(".csv"):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)

        df_clone = df.copy()

        # prepend the text
        df_clone["response"] = df_clone.apply(
            lambda row: f"Name of Founder: {row['Name']}\n\n{row['response']}",
            axis=1,
        )

        df_clone["setup"] = df_clone["setup"] + "_with_evalname"
        df = pd.concat([df, df_clone], ignore_index=True)

        # build new filename
        base, ext = os.path.splitext(file)
        new_filename = f"{base}_with_evalname{ext}"
        new_path = os.path.join(folder, new_filename)
        df.to_csv(new_path, index=False)
        print(f"Created {new_filename}")
