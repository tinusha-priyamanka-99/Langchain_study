import os
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from api import openai_api_key

os.environ["OPENAI_API_KEY"]= openai_api_key

chat = ChatOpenAI(
    temperature= 0.5,
    model= 'gpt-3.5-turbo'
)
system_message = SystemMessage(content= "You are an AI assistant")
human_message = HumanMessage(content= "answer for 5+4")

response = chat(
    [
        system_message,
        human_message
    ]
)

print(response)