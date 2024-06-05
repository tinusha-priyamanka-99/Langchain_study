from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone
import asyncio
from langchain_community.document_loaders import SitemapLoader

def get_website_data(sitemap_url):

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loader = SitemapLoader(
        sitemap_url
    )
    docs = loader.load()
    return docs

def split_data(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap = 200,
        length_function = len
    )

    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings