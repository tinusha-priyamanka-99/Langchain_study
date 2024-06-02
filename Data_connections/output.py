import re
import json
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
load_dotenv()

response_schemas = [
    ResponseSchema(name="question", description="Question generated from provided input text data."),
    ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma seperated."),
    ResponseSchema(name="answer", description="Correct answer for the asked question.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
print(output_parser)

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

chat_model = ChatOpenAI()
print(chat_model)

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            """When a text input is given by the user, please generate multiple choice questions
            from it along with the correct answer,
            \n{format_instructions}\n{user_prompt}
            """)
    ],
    input_variables=["user_prompt"],
    partial_variables={"format_instructions": format_instructions}
)

final_query = prompt.format_prompt(user_prompt = answer),
final_query.to