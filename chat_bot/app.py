import streamlit as st
from utils import get_website_data, split_data, create_embeddings

st.title("AI Assistance For Website")

st.sidebar.title("Enter the API Keys:")
st.session_state['HuggingFace_API_Key']= st.sidebar.text_input("HuggingFace API key:", 
                                                               type='password')
st.session_state['Pinecone_API_Key'] = st.sidebar.text_input("Pinecone API key", 
                                                             type='password')

load_button = st.sidebar.button("Load the data to Pinecone",
                                key="load_button")

if load_button:

    if st.session_state['HuggingFace_API_Key'] != "" and st.session_state['Pinecone_API_Key'] != "":
        
        site_data= get_website_data("https://www.langchain.com/sitemap.xml")
        st.write("Data pull done...")
        
        chunks_data = split_data(site_data)
        st.write("Splitting data done...")
        
        embeddings = create_embeddings(chunks_data)
        st.write("Embeddings instance creation done...")
        st.write("Pushing data to Pinecone done...")
        st.sidebar.write("Data pushed to Pinecone successfully!")

    else:
        st.sidebar.error("Please provide API keys...")
