from util import *
import openai
from openai import OpenAI
import requests
import pandas as pd
import os
import re
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# Prompt for Hybrid Identity narratives
startup_domains = [
    "Food & Beverage",
    "Health & Wellness",
    "Education & Training",
    "Fashion & Apparel",
    "Home & Living",
    "Agriculture",
    "Travel & Tourism",
    "Arts & Crafts",
    "Events & Entertainment",
    "Sports & Recreation",
    "Transportation",
    "Media & Publishing",
    "Beauty & Personal Care",
    "Construction & Real Estate",
    "Retail",
    "Nonprofit & Social Impact",
    "Environment & Sustainability",
    "Childcare & Parenting",
    "Pet Care",
    "Professional Services"
]

# Define all combinations of AI Usage and Social Idea
ai_social_combinations = [
    ("Yes", "Yes"),
    ("Yes", "No"),
    ("No", "Yes"),
    ("No", "No")
]

base_prompt = "Please generate a generic business idea and explain in 5 bullet points. Then, give information about the industry (in ISIC code), usage of AI (yes or no), " \
"and whether this idea is a social idea (yes or no). " \
"Please do it in the format: " \
"\n" \
"Business idea: (under 5 words)\n" \
"1. [Bullet Point 1]\n" \
"2. [Bullet Point 2]\n" \
"3. [Bullet Point 3]\n" \
"4. [Bullet Point 4]\n" \
"5. [Bullet Point 5]\n" \
"Industry: [Industry]\n" \
"AI Usage: [Yes/No]\n" \
"Social Idea: [Yes/No]"

def parse_response(content):
    # Extract business idea (title)
    idea_title_match = re.search(r"Business idea:\s*(.*)", content)
    idea_title = idea_title_match.group(1).strip() if idea_title_match else ""

    # Extract bullet points
    bullet_points = re.findall(r"\d+\.\s*(.*)", content)
    # Pad bullet points to always have 5 items
    bullet_points += [""] * (5 - len(bullet_points))

    # Extract industry, AI usage, and social idea
    industry_match = re.search(r"Industry:\s*(.*)", content)
    ai_usage_match = re.search(r"AI Usage:\s*(.*)", content)
    social_idea_match = re.search(r"Social Idea:\s*(.*)", content)

    return {
        "Business Idea": idea_title,
        "Bullet Point 1": bullet_points[0],
        "Bullet Point 2": bullet_points[1],
        "Bullet Point 3": bullet_points[2],
        "Bullet Point 4": bullet_points[3],
        "Bullet Point 5": bullet_points[4],
        "Industry": industry_match.group(1).strip() if industry_match else "",
        "AI Usage": ai_usage_match.group(1).strip() if ai_usage_match else "",
        "Social Idea": social_idea_match.group(1).strip() if social_idea_match else ""
    }

# Function to call OpenAI API with the hybrid prompt
# Collect responses in a dictionary
business_ideas = []
for domain in startup_domains:
    for ai_usage, social_idea in ai_social_combinations:
        # Add explicit instruction for AI Usage and Social Idea
        prompt = (
            base_prompt +
            f"\n\nThe business idea MUST have: AI Usage: {ai_usage}; Social Idea: {social_idea}."
        )
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

# Create DataFrame with all columns
df = pd.DataFrame(
    business_ideas,
    columns=[
        'Business Idea',
        'Bullet Point 1',
        'Bullet Point 2',
        'Bullet Point 3',
        'Bullet Point 4',
        'Bullet Point 5',
        'Industry',
        'AI Usage',
        'Social Idea'
    ]
)
print(df)

# Save DataFrame to CSV
df.to_csv('data/output/business_ideas2.csv', index=True)
