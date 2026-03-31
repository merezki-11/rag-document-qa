from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Pages loaded: {len(documents)}")
    if documents:
        print(f"Sample text from page 1: {documents[0].page_content[:300]}")

    print(f"Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    if len(chunks) == 0:
        print("ERROR: No chunks created. PDF may be image-based or empty.")
        return 0

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Done! Documents stored in chroma_db/")
    return len(chunks)

if __name__ == "__main__":
    ingest_pdf("attention.pdf")