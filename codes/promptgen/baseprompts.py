BUSINESSPROMPT = [
    "Generate a business pitch in 200 words length based on the following parameters:\n 1. Business idea: <BUSINESSIDEA>\n 2. Bullet Point 1: <BULLETPOINT1>\n 3. Bullet Point 2: <BULLETPOINT2>\n 4. Bullet Point 3: <BULLETPOINT3>\n 5. Bullet Point 4: <BULLETPOINT4>\n 6. Bullet Point 5: <BULLETPOINT5>\n",
    "Generate a 200 word business pitch based on the following information:\n 1. Business idea: <BUSINESSIDEA>\n 2. Bullet Point 1: <BULLETPOINT1>\n 3. Bullet Point 2: <BULLETPOINT2>\n 4. Bullet Point 3: <BULLETPOINT3>\n 5. Bullet Point 4: <BULLETPOINT4>\n 6. Bullet Point 5: <BULLETPOINT5>\n",
    "Create a 200 word business pitch using the following details:\n 1. Business idea: <BUSINESSIDEA>\n 2. Bullet Point 1: <BULLETPOINT1>\n 3. Bullet Point 2: <BULLETPOINT2>\n 4. Bullet Point 3: <BULLETPOINT3>\n 5. Bullet Point 4: <BULLETPOINT4>\n 6. Bullet Point 5: <BULLETPOINT5>\n",
    "Write a 200 word business pitch based on the following parameters:\n 1. Business idea: <BUSINESSIDEA>\n 2. Bullet Point 1: <BULLETPOINT1>\n 3. Bullet Point 2: <BULLETPOINT2>\n 4. Bullet Point 3: <BULLETPOINT3>\n 5. Bullet Point 4: <BULLETPOINT4>\n 6. Bullet Point 5: <BULLETPOINT5>\n",
    "Draft a 200 word business pitch using the following information:\n 1. Business idea: <BUSINESSIDEA>\n 2. Bullet Point 1: <BULLETPOINT1>\n 3. Bullet Point 2: <BULLETPOINT2>\n 4. Bullet Point 3: <BULLETPOINT3>\n 5. Bullet Point 4: <BULLETPOINT4>\n 6. Bullet Point 5: <BULLETPOINT5>\n"]

NAMEPROMPT = [
    "My name is <NAME>.",
    "I am called <NAME>.",
    "You can refer to me as <NAME>.",
    "I go by the name <NAME>.",
    "I am <NAME>."]

NONAME = [
    "Please no not mention my name.",
    "Do not include my name in the response.",
    "I do not want my name to be mentioned.",
    "Please exclude my name from the response.",
    "Kindly do not refer to me by name."
]


def replace_prompt_with_content(row):
    prompt = row['prompts']

    replacements = {
        '<BUSINESSIDEA>': row.get('Business Idea'),
        '<COUNTRY>': row.get('Bullet Point 1'),
        '<YEAR>': row.get('Bullet Point 2'),
        '<CURRENCY>': row.get('Bullet Point 3'),
        '<OBJECT>': row.get('Bullet Point 4'),
        '<UNIT>': row.get('Bullet Point 5'),
        '<NAME>': row.get('name'),
        '<UNIT2>': row.get('unit2'),
        '<CONVERSION>': row.get('conversion'),
        '<VALUE>': row.get('value'),
    }

    for placeholder, value in replacements.items():
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, str(value))

    return prompt

