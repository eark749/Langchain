import os 
from dotenv import load_dotenv
import streamlit as st
from langchain_core import output_parsers
load_dotenv(dotenv_path="/Users/vansh/Desktop/langchain/.env")
from langchain_community.llms import Ollama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant . Please  repsonse to the user queries"),
        ("user", "question: {question}")
    ]
)

def generate_response(question,llm, temperature, max_tokens,):
    llm = Ollama(model = llm)
    output_parsers = StrOutputParser()
    chain = prompt | llm | output_parsers
    answer = chain.invoke({"question":question})
    return answer


st.title("Enhanced Q&A Chatbot With OpenAI")


## Select the OpenAI model
llm=st.sidebar.selectbox("Select Open Source model",["llama2¸"])

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")



if user_input :
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")
