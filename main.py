import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated Import
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="SJCEM Campus Bot API - Pro RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for Lazy Loading
is_ai_loaded = False
advanced_retriever = None
llm = None
prompt_template = None

def load_ai_models():
    """Ye function tabhi chalega jab pehla message aayega, isse server deploy hone me timeout nahi hoga."""
    global is_ai_loaded, advanced_retriever, llm, prompt_template
    
    if is_ai_loaded:
        return # Agar pehle se load hai toh wapas load mat karo

    print("Loading Heavy AI Models in background... (This might take a minute)")
    
    # 1. Load Data
    DATA_FILE = "college_data_cleaned.md"
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()

    splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4 

    # 2. Chroma DB 
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="./chroma_db_updated", 
        embedding_function=embeddings
    )
    chroma_retriever = db.as_retriever(search_kwargs={"k": 4})

    advanced_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )
    
    # # 3. Ensemble
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    # )

    # # 4. Cross-Encoder Reranker
    # cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # compressor = CrossEncoderReranker(model=cross_encoder, top_n=6)

    # advanced_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, 
    #     base_retriever=ensemble_retriever
    # )

    # 5. LLM Setup
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

    prompt_template = PromptTemplate(
        template="""You are the official, intelligent AI Campus Assistant for St. John College of Engineering and Management (SJCEM), located in Village Vevoor, Manor Road, Palghar (East), under the Aldel Education Trust (NAAC A+ & NBA Accredited).

        YOUR CORE DIRECTIVE:
        1. PRIMARY SOURCE: Always prioritize the facts, tables, rules, and numbers provided in the 'Context' below.
        2. NEVER REFUSE OR SAY "I DON'T KNOW": You are strictly forbidden from saying "I do not have this information", "Not mentioned in the context", or "Please contact the admin". 
        3. SMART EXTRAPOLATION (When answer is missing from Context): If the user asks a question about SJCEM that is not explicitly covered in the Context, you MUST seamlessly generate a helpful, realistic, and confident answer by assuming standard practices of SJCEM, Mumbai University engineering colleges, and AICTE norms. 
        4. LANGUAGE MATCHING: Reply in the EXACT same language/style the user used. If the user asks in English, reply in English. If the user asks in Hinglish (Hindi written in Latin script), reply naturally in Hinglish.

        Previous Chat History:
        {history}

        Retrieved College Context: 
        {context}

        User Query: {question}

        Confident & Helpful Answer:""",
        input_variables=["history", "context", "question"]
    )
    
    is_ai_loaded = True
    print("AI Models Loaded Successfully!")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[ChatMessage] = []

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # API hit hote hi check karo models load hue hain ya nahi
    load_ai_models()
    
    recent_history = request.history[-6:]
    history_text = ""
    for msg in recent_history:
        history_text += f"{msg.role}: {msg.content}\n"

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
        
    return {"answer": final_text}

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")