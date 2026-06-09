from dotenv import load_dotenv
load_dotenv()

import base64
from groq import Groq

client = Groq()

with open("test.jpg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)