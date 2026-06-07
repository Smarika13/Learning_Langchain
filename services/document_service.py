from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_documents():
    loader=DirectoryLoader(
        "documents/",
        glob="**/*.txt",
        loader_cls=TextLoader
    )

    documents=loader.load()

    splitter=RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap=200

    )
    chunks=splitter.split_documents(documents)
    return chunks

#TestBlock
if __name__ == "__main__":
    chunks=load_and_chunk_documents()
    print(f"Total chunks: {len(chunks)}")
    print(chunks[0])