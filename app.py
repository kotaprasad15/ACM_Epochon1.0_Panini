from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from langchain_community.llms import LlamaCpp  # Corrected import here
from langchain.chains import RetrievalQA  # Corrected import here
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import json
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path to the local LLM model
local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Initialize LlamaCpp
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1
)
print("LLM Initialized....")

# Prompt template for the QA chain
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Qdrant client setup
url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

# Vectorstore database using Qdrant
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")

# Setup retriever
retriever = db.as_retriever(search_kwargs={"k": 1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Initialize the RetrievalQA chain
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    
    # Generate the response
    response = qa(query)
    print(response)
    
    # Extract details from the response
    answer = response["result"]
    source_document = response["source_documents"][0].page_content
    doc = response["source_documents"][0].metadata["source"]

    # Encode the response as JSON
    response_data = {
        "answer": answer,
        "source_document": source_document,
        "doc": doc
    }
    response_json = jsonable_encoder(response_data)

    return Response(json.dumps(response_json), media_type="application/json")
