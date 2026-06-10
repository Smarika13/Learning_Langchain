from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
import os


def create_embeddings(chunks):
    embedding_model=FastEmbedEmbeddings()
    if os.path.exists("faiss_index"):
       print("Loading FAISS index from disk...")
       vector_store = FAISS.load_local("faiss_index",embedding_model,allow_dangerous_deserialization=True)
    else:
        print("Building FAISS index from scratch...") 
        vector_store=FAISS.from_documents(chunks, embedding_model)
        vector_store.save_local("faiss_index")
        
    return vector_store

#TESTBLOCK
if __name__ == "__main__":
    from document_service import load_and_chunk_documents
    chunks=load_and_chunk_documents()
    vector_store=create_embeddings(chunks)
    print(f"Vector store created with {vector_store.index.ntotal}vectors")
