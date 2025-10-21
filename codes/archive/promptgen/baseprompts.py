NAMEPROMPT = [
    "Founder name: <NAME>"
    "Founder: <NAME>",
    "The founder's name is <NAME>."]



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

