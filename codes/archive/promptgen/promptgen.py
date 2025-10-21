import pandas as pd
import random
import openai
import os
import dotenv

# Call needed dataframes
surenames = pd.read_csv('data/output/surenames.csv')
busidea = pd.read_csv('data/output/business_ideas2.csv')

# Generate Prompt
