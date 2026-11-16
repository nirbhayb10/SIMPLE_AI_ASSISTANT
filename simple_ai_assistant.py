from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS, FUNCTION_NAME_TO_FUNCTION_MAP
import os
import json

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client and connect to Ollama
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

# Store complete conversation history
chat_history = []

# Define assistant behavior using a system prompt
chat_history.append(
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

    # Store user query in conversation memory
    chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # First API call
    # Model decides whether it needs to call a tool
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=chat_history,
        tools=TOOLS
    )

    msg = response.choices[0].message

    # Check if the model wants to use any tool
    if msg.tool_calls:
        chat_history.append(msg)    # Store assistant response in memory to maintain history

        for tool_call in msg.tool_calls:    # Execute each requested tool
            func_name = tool_call.function.name # Extract function name
            func_args = json.loads(tool_call.function.arguments or "{}")    # Extract function arguments
            
            fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]   # Get corresponding Python function
            result = fn(**func_args)    # Execute function with arguments

            # Save tool result into conversation history
            chat_history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                }
            )

    # Second API call
    # Sends tool output back to the model
    # so it can generate a final response
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=chat_history
    )

    msg = response.choices[0].message

    # Store assistant response in memory
    chat_history.append(
        {
            "role": "assistant",
            "content": msg.content
        }
    )

    # Display final response
    print("Assistant:", msg.content)

    # Debug: Print assistant role if needed
    # print(msg.role)