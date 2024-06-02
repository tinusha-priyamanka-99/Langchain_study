import os
import openai
from langchain_community.llms.huggingface_hub import HuggingFaceHub
import pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
load_dotenv()

def load_docs(directory):
    loader = PyPDFDirectoryLoader(directory)
    documents = loader.load()
    return documents

directory = 'D:\\Data Science\\Generative_AI\\Data_connections'
documents = load_docs(directory)
#print(len(documents))

def split_docs(documents,chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ) 
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)
print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name= 'all-MiniLM-L6-V2')

query_result = embeddings.embed_query('Hello query')
print(len(query_result))

pc = pinecone.Pinecone(
    api_key=os.environ.get("pinecone_api_key")
)
index_name= "mcq-creator"
index = pc.Index(index_name)

index = Pinecone.from_documents(docs,embeddings, index_name=index_name)

def get_similar_docs(query, k):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

llm = HuggingFaceHub(
    huggingfacehub_api_token= os.environ.get("huggingface_api"),
    repo_id="bigscience/bloom",
    model_kwargs={"temperature": 1e-10})

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    relavant_docs = get_similar_docs(query, k=1)
    #print(relavant_docs)
    response = chain.run(input_documents=relavant_docs,
                         question=query)
    return response

query_1 = "How is India's economy?"
answer = get_answer(query_1)
print(answer)
