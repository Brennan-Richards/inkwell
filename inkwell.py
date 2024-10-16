# This example requires the 'message_content' intent.

import discord

import os
import requests
import base64


def answer_question(question):
    PAGE_ONE_DOCUMENTATION = ""
    # Read in the page one docs from page_one_server_list.txt
    with open("page_one_inkwell_list.txt", "r") as file:
        PAGE_ONE_DOCUMENTATION = file.read()

    # Configuration
    API_KEY = "3c39f0eb0fa54a9f8cd5d54995e3a3f4"
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }

    # Payload for the request
    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": f"""You are an AI assistant named Inkwell that helps direct people towards the ways that they should leverage a discord server called Page One, which you are installed on.
                        Users will ask you questions about where they should post or interact with the discord server.
                        Answer users' questions based on the following guidelines which contain a list of all of the places to post on the server: 
                        { PAGE_ONE_DOCUMENTATION }
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

    ENDPOINT = "https://your-resource-name.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    
    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    # print(response.json())

    # Return the response as a dictionary
    return response.json()

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
    response = answer_question(message.content)
    # response = answer_question("Where should I post my story about femininity that flips femininity into a strength?")
    response_text = response['choices'][0]['message']['content']

    print("-" * 50)
    print(f"User message: {message.content}")

    # Debugging: Print the generated response
    print(f"Generated response: {response_text}")
    print("-" * 50)

    # Send the response to the channel
    await message.channel.send(response_text)


client.run('REDACTED-DISCORD-BOT-TOKEN')
# answer = answer_question("Where should I post my story about femininity that flips femininity into a strength?")
# response_text = answer['choices'][0]['message']['content']
# print(answer)
# print(type(answer))
# print(response_text)