from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from services.llm_service import get_llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question based only on the context below.
    If the answer is not in the context, say "I don't know".
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    
    llm = get_llm()
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain