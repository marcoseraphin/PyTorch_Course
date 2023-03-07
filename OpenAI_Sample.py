import os
import openai

from dotenv import load_dotenv

load_dotenv('TOKEN.env')  # take environment variables from .env.

OPEN_AI_API_KEY=os.getenv('OPEN_AI_API_KEY')

openai.api_key = os.getenv('OPEN_AI_API_KEY')

# Modelname see => https://platform.openai.com/docs/models/gpt-3-5
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Schreibe einen Bericht über Motorräder",
  temperature=0.4,
  max_tokens=1443,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(f"Response = {response}")
