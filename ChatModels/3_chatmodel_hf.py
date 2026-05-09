from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

client = InferenceClient(
    token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

messages = [
    {
        "role": "user",
        "content": "What is the capital of India?"
    }
]

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    max_tokens=100,
)

print(response.choices[0].message.content)