from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

def create_retriever(chunks, vector_store):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k=3

    faiss_retriever=vector_store.as_retriever(search_kwargs={"k":3})
    ensemble_retriever=EnsembleRetriever(
        retrievers = [bm25_retriever, faiss_retriever],
        weights = [0.5, 0.5]
    )
    return ensemble_retriever

if __name__ == "__main__":
    from document_service import load_and_chunk_documents
    from embedding_service import create_embeddings

    chunks = load_and_chunk_documents()
    vector_store = create_embeddings(chunks)
    retriever = create_retriever(chunks, vector_store)

    results = retriever.invoke("What is the capital of Nepal?")
    print(f"Total results: {len(results)}")
    print(results[0].page_content)