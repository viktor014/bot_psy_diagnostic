from g4f.client import Client


def response_gpt(answer):
    client = Client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": answer}],
    )
    return response.choices[0].message.content
