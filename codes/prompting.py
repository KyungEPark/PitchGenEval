from util import *
import openai
from openai import OpenAI
import requests
import pandas as pd
import os


client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)

# Prompt for Hybrid Identity narratives
prompt = "Please generate a "

# Example texts for testing - TODO: CALL real texts from the website
examples = [
    "Wir sind ein junges Gründerteam aus drei Personen. Nachdem jeder von uns seine ganz eigenen Erfahrung mit der Verschmutzung der Umwelt gemacht hat, kamen wir auf die Idee selbst Akzente zu setzen, um etwas aktiv dagegen zu tun. Für die Produktion greifen wir auf die Expertise eines exklusiven Partnernetzwerks zurück. Neben einem traditionsreichen deutschen Glashersteller, steht uns ein Unternehmen zur Seite, das sich durch mehr als 25 Jahre Erfahrung in der Produktion von Spezialgläsern auszeichnet.",
    "MELINA BUCHER Handtaschen vereinen Ethik mit Ästhetik. Handgefertigt in unserem Atelier in Süddeutschland, werden Jahrhunderte altes Täschnerhandwerk mit zukunftsweisenden Materialien kombiniert. Meisterliche Qualität trifft auf zirkuläres Design - für einzigartige Taschen im Einklang mit der Natur. ",
    "Der Starkmacher e.V. ist eine Art Ideen- und Projektbörse von und für Menschen, denen die Förderung und Entfaltung von Potentialen in Jugendlichen am Herzen liegt. Ziel ist es, ehrenamtliche und professionelle Kräfte in einer guten Zusammenarbeit zu bündeln, Netzwerke zu schaffen und auszubauen und wichtige Kompetenzen im Bereich der Jugend- und Bildungsarbeit wirksam auszuschöpfen. Die Projekte des Starkmacher e.V. werden von verschiedenen EU-Förderprogrammen unterstützt.",
    "Wir sind profitabel aber auch menschenfreundlich.",
    "Atemschutz ist ein Menschenrecht."
]

# Function to call OpenAI API with the hybrid prompt
# Collect responses in a dictionary
results = {f"Prompt {i+1}": [] for i in range(len(prompts))}

for i, prompt in enumerate(prompts):
    for example in examples:
        response = call_openai(prompt, example)
        results[f"Prompt {i+1}"].append(response)

# Create DataFrame: rows=examples, columns=prompts
df = pd.DataFrame(results, index=[f"Example {i+1}" for i in range(len(examples))])
df.to_csv('data/prompt_results.csv', index=True)
