import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = OpenAI(temperature=0, max_tokens=500)

folder_path = 'D:\\Data Science\\Generative_AI\\Resume reader'
loader = DirectoryLoader(folder_path, glob='*.txt', loader_cls=TextLoader)
documents = loader.load()

embeddings = HuggingFaceInstructEmbeddings()

prompt_template = """
Given the following context and a question, generate an answer using minimum number of words based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

questions = [
    "Mention the job title using less than four words.",
    "Mention the number of experience years needed for positions using less than six words.",
    "Mention the academic qualification using less than ten words."
]

all_results = []

for doc in documents:
    vectordb = FAISS.from_documents(
        documents=[doc],
        embedding=embeddings
    )

    retriever = vectordb.as_retriever()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Dictionary to store the results for this document
    results = {
        "Job title": [],
        "Experienced years": [],
        "Academic Qualification": []
    }

    # Query each question and store the results
    for question in questions:
        result = chain({"query": question})
        if question.startswith("Mention the job title"):
            results["Job title"].append(result['result'])
        elif question.startswith("Mention the number of experience years"):
            results["Experienced years"].append(result['result'])
        elif question.startswith("Mention the academic qualification"):
            results["Academic Qualification"].append(result['result'])

    # Add the results for this document to the list of all results
    all_results.append(results)

# Convert all results to a DataFrame
data = {
    "File": [],
    "Job title": [],
    "Experienced years": [],
    "Academic Qualification": []
}

for i, result in enumerate(all_results):
    data["File"].append(f'JD_{i+1}.txt')
    data["Job title"].append(result["Job title"])
    data["Experienced years"].append(result["Experienced years"])
    data["Academic Qualification"].append(result["Academic Qualification"])

df = pd.DataFrame(data)

print(df)

print("\nAcademic Qualification:")
for qualification in df["Academic Qualification"]:
    print(qualification)
