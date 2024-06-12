from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_openai import OpenAI
import pinecone
from langchain.vectorstores import Pinecone
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text

def split_data(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap = 200
    )
    docs = text_splitter.split_text(text)
    docs_chunks = text_splitter.create_documents(docs)
    return docs_chunks

def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings

def push_to_pinecone(pinecone_apikey,
                     pinecone_environment,
                     pinecone_index_name,
                     embeddings,
                     docs):

    pc = pinecone.Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment= pinecone_environment
    )

    index_name= pinecone_index_name
    index = Pinecone.from_documents(docs,embeddings,index_name=index_name)
    return index

#*****Functions for dealing with Model related tasks...*****

def read_data(data):
    df = pd.read_csv(data,delimiter=",", header=None)
    return df

def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-V2")
    return embeddings

def create_embeddings(df,embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

def split_train_test_data(df_sample):
    sentences_train, sentences_test, lables_train, lables_test = train_test_split(
    list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0)
    print(len(sentences_train))
    return sentences_train, sentences_test, lables_train, lables_test

def get_score(svm_classifier, sentences_test, labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score
