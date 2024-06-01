import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from streamlit_chat import message

load_dotenv()

# Initialize session state variables
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'API_Key' not in st.session_state:
    st.session_state.API_Key = ''

st.set_page_config(page_title="ChatGPT Clone", page_icon=":robot_face")
st.markdown("<h1 style='text-align: center;'>How can I assist you? </h>", unsafe_allow_html=True)

# Sidebar inputs
st.session_state.API_Key = st.sidebar.text_input("What's your API key", type="password")
summarise_button = st.sidebar.button("Summarise the conversation", key="Summarise")
if summarise_button:
    summarise_placeholder = st.sidebar.write("Hello friend")

def getresponse(userInput, api_key):
    if st.session_state.conversation is None:
        llm = ChatOpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo'
        )
        # Create a conversation chain with memory
        st.session_state.conversation = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationBufferMemory()
        )

    # Interact with the conversation chain
    response = st.session_state.conversation.predict(input=userInput)
    print(st.session_state.conversation.memory.buffer)

    return response

response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Your question goes here:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

        if submit_button:
            st.session_state.messages.append(user_input)
            model_response = getresponse(user_input, st.session_state.API_Key)
            st.session_state.messages.append(model_response)

            with response_container:
                for i in range(len(st.session_state.messages)):
                    if (i % 2) == 0:
                        message(st.session_state.messages[i], is_user=True, key=str(i) + '_user')
                    else:
                        message(st.session_state.messages[i], key=str(i) + '_AI')
