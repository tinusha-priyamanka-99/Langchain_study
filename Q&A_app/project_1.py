import streamlit as st
import requests
import os
from api import openai_api_key

os.environ['OPENAI_API_KEY'] = openai_api_key

def load_answer(question):
    headers = {
        'Authorization': f'Bearer {os.environ["OPENAI_API_KEY"]}',
        'Content-Type': 'application/json'
    }

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
    response_json = response.json()
    return response_json['choices'][0]['message']['content']

st.set_page_config(page_title="LangChain Demo",
                   page_icon=":robot:")
st.header("LangChain Demo")

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()
submit = st.button('Generate')

if submit and user_input:
    response = load_answer(user_input)
    st.subheader('Answer:')
    st.write(response)
