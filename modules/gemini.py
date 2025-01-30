import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    # model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def conversational_prompt(
    messages: List[Dict[str, str]],
    system_prompt: str = "You are a helpful conversational assistant. Respond in a short, concise, friendly manner.",
) -> str:
    """
    Send a conversational prompt to Gemini with message history.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        str: The model's response
    """
    try:
        chat = model.start_chat(history=[])

        # Add messages to the chat
        for message in messages:
            if message["role"] == "user":
                chat.send_message(message["content"])
            elif message["role"] == "assistant":
                chat.send_message(message["content"])

        # Get the final response
        response = chat.last.text
        return response
    except Exception as e:
        raise Exception(f"Error in conversational prompt: {str(e)}")

def prompt(prompt_text: str) -> str:
    """
    Send a prompt to Gemini.

    Args:
        prompt_text: The prompt text.

    Returns:
        The response from Gemini.
    """
    try:
        messages = [{"role": "user", "content": prompt_text}]
        response = conversational_prompt(messages)
        return response
    except Exception as e:
        raise Exception(f"Error in prompt: {str(e)}")
