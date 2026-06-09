from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

print("KEY FOUND =", bool(os.getenv("GROQ_API_KEY")))

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "say hello"}
    ]
)

print(response.choices[0].message.content)