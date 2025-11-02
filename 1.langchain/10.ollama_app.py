import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(dotenv_path="/Users/vansh/Desktop/langchain/.env")

## lagnchain tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

#prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant. please respond to the questions asked"),
    ("user","Question:{question}")
])

# streamlit framework
st.title("langchain demo with ollama")
input_text = st.text_input("what question u have in mind")

#ollama model
llm = Ollama(model="llama2")
output = StrOutputParser()
chain = prompt | llm | output

if input_text:
    st.write(chain.invoke({"question": input_text}))
