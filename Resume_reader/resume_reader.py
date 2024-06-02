import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = OpenAI(temperature=0, max_tokens=500)

folder_path = 'D:\\Data Science\\Generative_AI\\Resume_reader'
loader = DirectoryLoader(folder_path, glob='*.pdf', loader_cls=PyMuPDFLoader)
documents = loader.load()

# Print the loaded documents for debugging
print(f"Number of documents loaded: {len(documents)}")
for doc in documents:
    print(f"Document metadata: {doc.metadata}")

embeddings = HuggingFaceInstructEmbeddings()

prompt_template = """
You are an resume information extractor.Given the following context and a question, 
generate an answer using minimum number of words based on this context only.
If there are more answers for a question, provide a list of those answers.
If the answer is not found in the context, state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

questions = [
    "Mention the job field.",
    "Mention the number of experienced years of each positions as a list.",
    "Mention the academic qualifications as a list."
]

all_results = []

# Use a set to keep track of processed file names
processed_files = set()

for doc in documents:
    file_name_with_path = doc.metadata["source"]
    file_name = os.path.basename(file_name_with_path)
    if file_name in processed_files:
        continue  # Skip if already processed
    processed_files.add(file_name)
    
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
    results["File"] = file_name  # Store the file name without the path
    all_results.append(results)

# Convert all results to a DataFrame
data = {
    "File": [],
    "Job title": [],
    "Experienced years": [],
    "Academic Qualification": []
}

for result in all_results:
    data["File"].append(result["File"])
    data["Job title"].append(result["Job title"])
    data["Experienced years"].append(result["Experienced years"])
    data["Academic Qualification"].append(result["Academic Qualification"])

df = pd.DataFrame(data)

print(df)

excel_file_path = "output_pdf2.xlsx"
df.to_excel(excel_file_path, index=False)

#print("\nAcademic Qualification:")
#for qualification in df["Academic Qualification"]:
#    print(qualification)
