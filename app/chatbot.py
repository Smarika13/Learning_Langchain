from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from services.llm_service import get_llm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question based only on the context below.
        If the answer is not in the context, say "I don't know"
        
        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    llm = get_llm()
    
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain