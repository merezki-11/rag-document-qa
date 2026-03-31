from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import shutil
import tempfile

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIR = "./chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_vectorstore():
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the question based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {input}
""")
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 4}),
        combine_docs_chain
    )

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Document QA API is running!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    vectorstore = get_vectorstore()
    qa_chain = get_qa_chain(vectorstore)
    result = qa_chain.invoke({"input": request.question})
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["context"]]
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise HTTPException(status_code=400, detail="PDF appears to be empty or image-based")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

        return {
            "message": f"Successfully processed {file.filename}",
            "chunks": len(chunks),
            "pages": len(documents)
        }
    finally:
        os.unlink(tmp_path)