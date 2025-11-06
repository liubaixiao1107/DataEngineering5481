from openai import OpenAI
client = OpenAI(api_key="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")

response = client.responses.create(
    model="gpt-5",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Analyze the letter and provide a summary of the key points.",
                },
                {
                    "type": "input_file",
                    "file_url": "cleaned",
                },
            ],
        },
    ]
)

print(response.output_text)

