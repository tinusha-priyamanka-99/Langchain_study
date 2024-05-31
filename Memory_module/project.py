import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (ConversationBufferMemory,
                              ConversationSummaryMemory)

load_dotenv()

if 'conversation' not in st.session_state:
    st.session_state['conversation']=None

st.set_page_config(page_title="ChatGPT Clone",
                   page_icon=":robot_face")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h>",
            unsafe_allow_html=True)

api_key = st.sidebar.text_input("What's your API key",
                                type="password")
summarise_button = st.sidebar.button("Summarise the conversation",
                                     key="Summarise")
if summarise_button:
    summarise_placeholder = st.sidebar.write("Hello friend")

def getresponse(userInput):

    if st.session_state['conversation'] is None:

        llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo'
        )

        # Create a conversation chain with memory
        st.session_state = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory()
        )

# Interact with the conversation chain
    response=st.session_state.predict(input= userInput)
    print(st.session_state.memory.buffer)


    return response

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form',
                 clear_on_submit=True):
        user_input = st.text_area("Your question goes here:",
                                  key = 'input',
                                  height=100)
        submit_button = st.form_submit_button(label='send')
        if submit_button:
            answer = getresponse(user_input)
            with response_container:
                st.write(answer)



