from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import fitz
from langchain_core.documents import Document

from config import PDF_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL


def build_vectorstore():

    print("Loading textbook...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()

    print("Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print("Creating embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Building FAISS index...")

    vector_db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_db.save_local(VECTOR_DB_PATH)

    print("Vector store created successfully!")


if __name__ == "__main__":
    build_vectorstore()