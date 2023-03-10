import os
import openai
from dotenv import load_dotenv

load_dotenv('TOKEN.env')  # take environment variables from .env.

OPEN_AI_API_KEY=os.getenv('OPEN_AI_API_KEY')
openai.api_key=OPEN_AI_API_KEY
#openai.organization = os.getenv("OPENAI_ORGANIZATION") 

conversation=[{"role": "system", "content": "You are a helpful assistant."}]

while(True):
    user_input = input("Frage: ")     
    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = conversation,
        temperature=2,
        max_tokens=250,
        top_p=0.9
    )

    conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
    print("\n" + response['choices'][0]['message']['content'] + "\n")