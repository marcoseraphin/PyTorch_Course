import os
import openai

from dotenv import load_dotenv

load_dotenv('TOKEN.env')  # take environment variables from .env.

OPEN_AI_API_KEY=os.getenv('OPEN_AI_API_KEY')
openai.api_key=OPEN_AI_API_KEY

# Modelname see => https://platform.openai.com/docs/models/gpt-3-5
response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Welcher Laptop wäre ideal für Office Arbeiten in Bezug auf Speicher und Chipsatz",
  temperature=0.4,
  max_tokens=1443,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(f"Response = {response}")
print(f"Text = {response.choices[0].text}")
