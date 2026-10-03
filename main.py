from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List

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

# API Key Setup
os.environ["GROQ_API_KEY"] = "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"

app = FastAPI(title="SJCEM Campus Bot API - Pro RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up Advanced RAG Pipeline...")

# 1. Sahi File Load Karo aur Chunk Size Badhao (Tables aur Rules intact rakhne ke liye)
DATA_FILE = "college_data_cleaned.md" # Make sure yehi file folder me ho
loader = TextLoader(DATA_FILE, encoding="utf-8")
docs = loader.load()

# 1200 chunk size se poori table aur rules ek sath aayenge
splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Initialize BM25 Retriever (Keyword Search)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 8 

# 2. ChromaDB ko Chunks ke sath Sync Karo (Semantic Search)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# from_documents use karne se naya data hamesha index hoga
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db_updated"
)
chroma_retriever = db.as_retriever(search_kwargs={"k": 8})

# 3. Combine both into an Ensemble Retriever (Hybrid Search)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
)

# 4. Cross-Encoder Reranker (Top 6 chunks bhejenge taaki koi detail miss na ho)
print("Loading Cross-Encoder Reranker...")
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=6)

advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)

# Temperature 0.4 rakha hai taaki missing info par smartly answer bana sake
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

# 5. SMART CONTEXTUAL ASSUMPTION PROMPT (No "I don't know" allowed)
prompt_template = PromptTemplate(
    template="""You are the official, intelligent AI Campus Assistant for St. John College of Engineering and Management (SJCEM), located in Village Vevoor, Manor Road, Palghar (East), under the Aldel Education Trust (NAAC A+ & NBA Accredited).

    YOUR CORE DIRECTIVE:
    1. PRIMARY SOURCE: Always prioritize the facts, tables, rules, and numbers provided in the 'Context' below.
    2. NEVER REFUSE OR SAY "I DON'T KNOW": You are strictly forbidden from saying "I do not have this information", "Not mentioned in the context", or "Please contact the admin". 
    3. SMART EXTRAPOLATION (When answer is missing from Context): If the user asks a question about SJCEM that is not explicitly covered in the Context, you MUST seamlessly generate a helpful, realistic, and confident answer by assuming standard practices of SJCEM, Mumbai University engineering colleges, and AICTE norms. 
       - Stay strictly within the realistic boundaries of SJCEM (Departments: AI/ML, Comps, IT, EXTC, Data Science, Mechanical, Civil; Portals: Aldel MIS; Location: Palghar).
       - Do NOT invent absurd claims (like having an airport on campus) and avoid inventing fake human names for specific posts if unknown—instead, refer to the respective designation/office confidently (e.g., "The Head of the AI/ML Department and faculty coordinators located on the 2nd Floor, A-Wing oversee this...").
    4. LANGUAGE MATCHING: Reply in the EXACT same language/style the user used. If the user asks in English, reply in English. If the user asks in Hinglish (Hindi written in Latin script), reply naturally in Hinglish.

    Previous Chat History:
    {history}

    Retrieved College Context: 
    {context}

    User Query: {question}

    Confident & Helpful Answer:""",
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
    # Sirf last 6 messages bhejo taaki context window clean rahe
    recent_history = request.history[-6:]
    history_text = ""
    for msg in recent_history:
        history_text += f"{msg.role}: {msg.content}\n"

    # Retrieve top relevant chunks
    docs = advanced_retriever.invoke(request.query)
    retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    final_prompt = prompt_template.format(
        history=history_text,
        context=retrieved_context, 
        question=request.query
    )
    
    response = llm.invoke(final_prompt)
    
    raw_content = response.content
    final_text = str(raw_content) if raw_content else "Welcome to SJCEM! Please let me know what specific academic or campus detail you need."
        
    return {
        "answer": final_text
    }