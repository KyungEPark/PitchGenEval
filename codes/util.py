# OpenAI calling
def call_openai(prompt, text, model="gpt-3.5-turbo",  max_tokens=1000):
    import openai
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a start-up expert."},
                  {"role": "user", "content": prompt}
                  ],
        max_tokens=max_tokens,
        temperature = 0.7,
    )
    return response.choices[0].message['content']
