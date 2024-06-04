from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_openai import OpenAI
 
def query_agent(data,query):
    df = pd.read_csv(data,engine='python')

    llm = OpenAI()

    agent = create_pandas_dataframe_agent(llm, df, verbose=True,encoding = 'utf-8')

    return agent.run(query)