import pinecone
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores.pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback
import os
import joblib

def pull_from_pinecone(pinecone_api_key,
                       pinecone_environment,
                       pinecone_index_name,
                       embeddings):
    pc = pinecone.Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment= pinecone_environment
    )

    index_name= pinecone_index_name
    index = Pinecone.from_existing_index(index_name,embeddings)
    return index

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings

def get_similar_docs(index,query,k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(docs,user_input):
    chain = load_qa_chain(OpenAI(), chain_type='stuff')
    with get_openai_callback() as cb:
        response = chain.run(input_documents = docs,
                                question=user_input)
    return response

def predict(query_result):
    Fitmodel = joblib.load('modelsvm.pkl')
    result=Fitmodel.predict([query_result])
    return result[0]
