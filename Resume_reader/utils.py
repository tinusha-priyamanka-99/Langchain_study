import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
import pinecone
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.llms import HuggingFaceHub

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def creatw_docs(user_pdf_list, unique_id):
    docs=[]
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)

        docs.append(document(
            page_content = chunks,
            metadata={"name": filename.name,
                      "id": filename.id,
                      "type":filename.type,
                      "size": filename.size,
                      "unique_id":unique_id},
        ))
    return docs
    