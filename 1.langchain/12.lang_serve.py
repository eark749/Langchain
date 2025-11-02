from cgitb import text
from fastapi import FastAPI
import fastapi
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os 
from dotenv import load_dotenv
from langserve import add_routes
load_dotenv(dotenv_path="/Users/vansh/Desktop/langchain/.env")

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)


system_template = "translate the follwoing into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}")]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

app = FastAPI(title="langchain server", version="1.0", description="a simple api server using langchain runnable interfaces")

add_routes (
    app,
    chain,
    path = "/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)