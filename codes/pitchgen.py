import pandas as pd
import random
import openai
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()    
# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load prompts


# Call response
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": "You are a start-up expert in the domain of " + domain},
        {"role": "user", "content": prompt}
    ],
    max_completion_tokens=5000
)
# Extract the content from the response
content = response.choices[0].message.content
parsed = parse_response(content)
business_ideas.append(parsed) 

# Save DataFrame to CSV
df.to_csv('data/output/business_ideas2.csv', index=True)
