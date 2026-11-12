from openai import OpenAI
from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Create OpenAI client using Ollama endpoint
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

# Store complete conversation history
messages = []

# Define assistant behavior using a system prompt
messages.append(
    {
        "role": "system",
        "content": "Act as humble agent. Always answer in less than 20 words. Never exceed 20 words."
    }
)

# Main conversation loop
while True:

    # Take user input
    user_input = input("\nNirbhay: ")

    # Exit conditions
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    # Store user's message in conversation memory
    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Send entire conversation history to the model
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages
    )

    # Store assistant response in memory
    messages.append(
        {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
    )

    # Display assistant response
    print("Assistant:", response.choices[0].message.content)

    # Debug: Print assistant role if needed
    # print(response.choices[0].message.role)