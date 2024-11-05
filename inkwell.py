# This example requires the 'message_content' intent.

import discord
import os
import aiohttp
import asyncio
import logging
import PyPDF2
# Solve environment variables appearing unset
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

def load_page_one_documentation():

    # Read in the page one docs from page_one_inkwell_list.txt
    # with open("page_one_inkwell_list.txt", "r") as file:
    #     page_one_documentation = file.read()

    page_one_documentation = ""
    # Read in all the PDFs in the 'documentation' directory and add their contents to the page_one_documentation
    for filename in os.listdir("documentation"):
        if filename.endswith(".pdf"):
            with open(f"documentation/{filename}", "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_number in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_number]
                    _docs = page.extract_text()
                    # Remove all newlines and replace them with spaces
                    _docs = _docs.replace("\n", " ")
                    page_one_documentation += _docs

    # Write the page_one_documentation to a TXT file
    # with open("page-one-docs-11042024.txt", "w") as file:
    #     file.write(page_one_documentation)

    return page_one_documentation

PAGE_ONE_DOCUMENTATION = load_page_one_documentation()

# Configuration
endpoint = "https://your-resource-name.openai.azure.com/"
deployment_name = "gpt-4o"  # Ensure this matches your deployment name in Azure
api_version = "2024-08-01-preview"  # Verify this is the correct API version

# It's safer to use environment variables for API keys and tokens
API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

async def answer_question(question):
    # Payload for the request
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an AI assistant named Inkwell that helps direct people towards the ways that they should leverage a Discord server called Page One, which you are installed on.
Users will ask you questions about where they should post or interact with the Discord server.
Answer users' questions based on the following guidelines, which contain a list of all the places to post on the server:
{PAGE_ONE_DOCUMENTATION}
                        """
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 800
    }

    endpoint_url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"

    # Send request
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_message = await response.text()
                    logging.error(f"HTTP Error {response.status}: {error_message}")
                    # Optionally, parse the error message to provide more details
                    return "I'm sorry, but I'm having trouble processing your request right now. This may go against my policies."
                result = await response.json()
                return result['choices'][0]['message']['content']
    except Exception as e:
        logging.exception("An unexpected error occurred.")
        return "An unexpected error occurred. Please try again later."

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Use the OpenAI API to generate a response
    response_text = await answer_question(message.content)

    print("-" * 50)
    print(f"User message: {message.content}")

    # Debugging: Print the generated response
    print(f"Generated response: {response_text}")
    print("-" * 50)

    # Send the response to the channel
    await message.channel.send(response_text)

# Use environment variable for Discord bot token
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is not set.")

client.run(DISCORD_BOT_TOKEN)