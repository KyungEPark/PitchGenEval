import pandas as pd
import random
from openai import OpenAI
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()    
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Load busideas
busidea = pd.read_csv('data/raw/busidea.csv')

# Create a list to store pitches
pitches = []

for index, row in busidea.iterrows():
    domain = row['category']
    idea = row['idea']

    prompt = f"Generate a detailed business pitch for a startup in the {domain} domain based on the idea: {idea}. Make it within 500 words."

    # Call response
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": f"You are a start-up expert in the domain of {domain}."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    content = response.choices[0].message.content
    pitches.append(content)

# Add pitches to the DataFrame
busidea['pitch'] = pitches

# Save DataFrame to CSV
busidea.to_csv('data/output/business_pitches.csv', index=True)
