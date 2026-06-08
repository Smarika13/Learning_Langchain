from fastapi import FastAPI
from pydantic import BaseModel
from services.document_service import load_and_chunk_documents
from services.embedding_service import create_embeddings
from services.retriever_service import create_retriever
from app.chatbot import create_rag_chain

app=FastAPI()

chunks = load_and_chunk_documents()
vector_store = create_embeddings(chunks)
retriever = create_retriever(chunks, vector_store)
rag_chain = create_rag_chain(retriever)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request:ChatRequest):
    response = rag_chain.invoke(request.question)
    return {"answer":response}