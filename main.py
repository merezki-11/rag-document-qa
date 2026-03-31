from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the question based only on the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {input}
""")

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
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
    result = retrieval_chain.invoke({"input": request.question})
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["context"]]
    }