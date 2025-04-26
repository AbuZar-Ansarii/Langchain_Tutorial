from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# --- Set API Key ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCV9Am8hcdsJkGYnhB7XjtCiOu_ZkQjjqM"

# --- FastAPI App ---
app = FastAPI()

# --- Paths ---
PDF_PATH = os.path.join("C:/", "Users", "abu", "PycharmProjects", "pythonProject", "Charaka-Samhita-Acharya-Charaka.pdf")
DB_PATH = os.path.join("C:/", "Users", "abu", "PycharmProjects", "pythonProject", "db")

# --- Load or Create Vector DB ---
if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_PATH)
    vectordb.persist()
else:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# --- Gemini LLM ---
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# --- Prompt ---
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful Ayurveda assistant. Answer the user's question only using the provided context.
If the answer is not found in the context, respond with:
"Information not available in provided context."

Context:
{context}

Question:
{question}

Answer:
"""
)

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# --- Request Body Schema ---
class Query(BaseModel):
    question: str

# --- API Endpoint ---
@app.post("/ask")
async def ask_question(query: Query):
    response = qa_chain({"query": query.question})
    return {
        "answer": response["result"],
        "sources": [doc.metadata.get("source", "N/A") for doc in response["source_documents"]]
    }

print("API is running successfully 🚀")

