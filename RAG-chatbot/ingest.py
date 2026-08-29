import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS


def process_pdf(file_path: str) -> FAISS:
    """Load a PDF, split into chunks, embed with Ollama, and store in a FAISS index.

    Returns:
        FAISS vector store containing the embedded document chunks.
    """
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)

    # Create embeddings using Ollama model (default llama2)
    model_name = os.getenv('OLLAMA_MODEL', 'llama2')
    embeddings = OllamaEmbeddings(model=model_name)

    # Build FAISS index
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store
