import streamlit as st
from dotenv import load_dotenv
from user_utils import *

if 'HR_tickets' not in st.session_state:
    st.session_state['HR_tickets']= []
if 'IT_tickets' not in st.session_state:
    st.session_state['IT_tickets']=[]
if 'Transport_tickets' not in st.session_state:
    st.session_state['Transport_tickets']= []

def main():
    load_dotenv()

    st.header("Automatic Ticket Classification Tool")
    st.write("Please ask your question:")
    user_input = st.text_input("Search")

    if user_input:

        embeddings = create_embeddings()

        index = pull_from_pinecone(os.environ.get("PINECONE_API_KEY"),
                                   "us-east-1",
                                   "ticket-generator",
                                   embeddings)
        relavant_docs = get_similar_docs(index,user_input)

        response = get_answer(relavant_docs,user_input)
        st.write(response)

        button = st.button("Submit ticket?")

        if button:
            
            embeddings= create_embeddings()
            query_result = embeddings.embed_query(user_input)

            department_value = predict(query_result)
            st.write("Your ticket has been submitted to : "+ department_value)

            if department_value == "HR":
                st.session_state["HR_tickets"].append(user_input)
            elif department_value == "IT":
                st.session_state["IT_tickets"].append(user_input)
            else: 
                st.session_state["Transport_tickets"].append(user_input)
        
        

if __name__ == "__main__":
    main()