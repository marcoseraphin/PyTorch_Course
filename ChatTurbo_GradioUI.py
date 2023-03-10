import gradio as gr
import os
import openai, subprocess
from dotenv import load_dotenv

load_dotenv('TOKEN.env')  # take environment variables from .env.

OPEN_AI_API_KEY=os.getenv('OPEN_AI_API_KEY')
openai.api_key=OPEN_AI_API_KEY

messages = [{"role": "system", "content": 'Du bist ein Computer Experte'}]

def transcribe(audio):
    global messages

    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response["choices"][0]["message"]
    messages.append(system_message)

    subprocess.call(["say", system_message['content']])

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch()
ui.launch()