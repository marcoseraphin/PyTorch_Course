import os
from dotenv import load_dotenv
from aleph_alpha_client import Client, CompletionRequest, Prompt
from dotenv import load_dotenv

load_dotenv('AA_TOKEN.env')  # take environment variables from .env.

AA_TOKEN=os.getenv('AA_TOKEN')

client = Client(token=os.getenv('AA_TOKEN'))
request = CompletionRequest(
    prompt=Prompt.from_text("Schreibe einen Bericht über Motorräder"),
    maximum_tokens=64,
)
response = client.complete(request, model="luminous-extended")

print(response.completions[0].completion)