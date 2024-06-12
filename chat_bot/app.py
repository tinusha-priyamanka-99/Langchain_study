import streamlit as st
from utils import *
import constants

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
         
        site_data= get_website_data(constants.WEBSITE_URL)
        st.write("Data pull done...")
        
        chunks_data = split_data(site_data)
        st.write("Splitting data done...")
        
        embeddings = create_embeddings()
        st.write("Embeddings instance creation done...")

        push_to_pinecone(st.session_state['Pinecone_API_Key'],
                         constants.PINECONE_ENVIRONMENT,
                         constants.PINECONE_INDEX,
                         embeddings,
                         chunks_data)
        st.write("Pushing data to Pinecone done...")
        st.sidebar.write("Data pushed to Pinecone successfully!")



prompt = st.text_input('How can I help you?', key='prompt')
document_count = st.slider('No of links to return - (0 LOW || 5 HIGH)',0,5,2,step=1)

submit = st.button('Search')

if submit:
        
        if st.session_state['HuggingFace_API_Key'] != "" and st.session_state['Pinecone_API_Key'] != "":
             
             embeddings = create_embeddings()
             st.write("Embeddings instance creation done...")

             index = pull_from_pinecone(st.session_state['Pinecone_API_Key'],
                         constants.PINECONE_ENVIRONMENT,
                         constants.PINECONE_INDEX,
                         embeddings)
             st.write("Pinecone index retrieval done...")

             relavant_docs = get_similar_docs(index,prompt,document_count)   
             st.write(relavant_docs)

             st.success('Please find the search results :')


             for document in relavant_docs:
                  
                  st.write("Results : "+ str(relavant_docs.index(document)+1)+ "**")
                  st.write("**Info**: "+document.page_content)
                  st.write("**Link**: "+ document.metadata['source'])

else:
        st.sidebar.error("Please provide API keys...")
