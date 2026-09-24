from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List

# Naye Imports Advanced RAG ke liye
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Yahan apni Groq API Key daalna
os.environ["GROQ_API_KEY"] = "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"

app = FastAPI(title="College Bot API - Advanced RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up Advanced RAG Pipeline...")

# 1. Load Raw Docs for BM25 (Keyword Search)
loader = TextLoader("college_data.md", encoding="utf-8")
docs = loader.load()
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Initialize BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5 # Keyword match se top 5 nikalega

# 2. Load ChromaDB for Vector Search (Semantic Match)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
chroma_retriever = db.as_retriever(search_kwargs={"k": 5}) # Vector match se top 5 nikalega

# 3. Combine both into an Ensemble Retriever (Hybrid Search)
# Ye dono result ko combine karega 50-50 weightage ke sath
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 4. Add Cross-Encoder Reranker (The Judge)
# Ye model query aur combined chunks ko re-evaluate karega aur strictly
print("Loading Cross-Encoder Reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3) # Final sirf top 3 chunks LLM ko jayenge

# Final Advanced Retriever pipeline
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)

# Load LLM
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)

# Strict Prompt Template
prompt_template = PromptTemplate(
    template="""You are a strict, helpful, and student-friendly AI assistant for a college campus.
    Your main purpose is to help students, faculty, and visitors with college-related questions such as academics, departments, courses, examinations, admissions, fees, campus facilities, events, placements, library, timetable, attendance, rules, clubs, laboratories, and general student guidance.
    Use the provided Context as the primary source. If the Context does not contain enough information, provide useful general guidance while clearly distinguishing it from confirmed college-specific information. 
    If the answer is not in the Context, explicitly say 'I do not have this information, please contact the admin'. DO NOT guess or hallucinate.
    CRITICAL: You must generate your final response entirely in the exact language the user used in their original query if it ask query in hinglish language then you have to provide answer in hinglish language and so on.

    Previous Chat History:
    {history}

    Context: 
    {context}

    User Query: {question}

    Answer:""",
    input_variables=["history", "context", "question"]
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    history_text = ""
    for msg in request.history:
        history_text += f"{msg.role}: {msg.content}\n"

    # YAHAN ADVANCED RETRIEVER USE HOGA
    docs = advanced_retriever.invoke(request.query)
    retrieved_context = "\n\n".join([doc.page_content for doc in docs])
    
    final_prompt = prompt_template.format(
        history=history_text,
        context=retrieved_context, 
        question=request.query
    )
    
    response = llm.invoke(final_prompt)
    
    raw_content = response.content
    final_text = str(raw_content) if raw_content else "Sorry, I could not generate a response."
        
    return {
        "answer": final_text
    }