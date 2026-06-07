from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings


def create_embeddings(chunks):
    embedding_model=FastEmbedEmbeddings()
    vector_store=FAISS.from_documents(chunks, embedding_model)
    return vector_store

#TESTBLOCK
if __name__ == "__main__":
    from document_service import load_and_chunk_documents
    chunks=load_and_chunk_documents()
    vector_store=create_embeddings(chunks)
    print(f"Vector store created with {vector_store.index.ntotal}vectors")
