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
print(f"text-davinci-003 => Response = {response}")
print(f"text-davinci-003 => Text = {response.choices[0].text}")

response2 = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  prompt="Welcher Laptop wäre ideal für Office Arbeiten in Bezug auf Speicher und Chipsatz",
  temperature=0.4,
  max_tokens=1443,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(f"gpt-3.5-turbo => Response = {response2}")
print(f"gpt-3.5-turbo => Text = {response2.choices[0].text}")
