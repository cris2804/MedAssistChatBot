from groq import Groq
import os
from apikey import groq_apikey

os.environ['GROQ_API_KEY']= groq_apikey

import os
from groq import Groq


# IMAGE_DATA_URL="https://img.freepik.com/free-photo/blooming-purple-flowers_23-2147836282.jpg"
IMAGE_DATA_URL="https://images.unsplash.com/photo-1486365227551-f3f90034a57c"

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "what you see"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": IMAGE_DATA_URL
                    }
                }
            ]
        }
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message)