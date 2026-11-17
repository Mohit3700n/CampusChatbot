# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import MarkdownTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# def build_vector_db():
#     print("Loading data...")
#     # 1. Load the Markdown file
#     loader = TextLoader("college_data_cleaned.md", encoding="utf-8")
#     docs = loader.load()

#     # 2. Split into chunks
#     splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)
#     print(f"Total chunks created: {len(chunks)}")

#     # 3. Use Local Open-Source Embeddings (Fast, Free, No API Key needed)
#     print("Loading local embedding model (Pehli baar thoda time lega download hone me)...")
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     # 4. Save to ChromaDB
#     print("Saving to ChromaDB...")
#     db = Chroma.from_documents(
#         documents=chunks, 
#         embedding=embeddings, 
#         persist_directory="./chroma_db"
#     )
    
#     print("✅ Data successfully saved to local ChromaDB!")

# if __name__ == "__main__":
#     build_vector_db()

# import os
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import MarkdownTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore

# # Apni actual keys yahan daalo (Sirf ek baar upload karne ke liye)


# print("Loading College Data...")
# loader = TextLoader("college_data_cleaned.md", encoding="utf-8")
# docs = loader.load()

# splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
# chunks = splitter.split_documents(docs)

# print("Generating Embeddings and Uploading to Pinecone Cloud (1-2 minutes wait)...")
# # Exact updated line
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

# # Ye data ko Cloud par bhej dega
# PineconeVectorStore.from_documents(
#     chunks, 
#     embeddings, 
#     index_name="sjcem-chatbot"
# )
# print("✅ Success! Data is now safe in Cloud DB.")



import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not google_api_key or not pinecone_api_key:
    raise ValueError("GOOGLE_API_KEY and PINECONE_API_KEY must be configured.")

print("Loading College Data...")
loader = TextLoader("college_data_cleaned.md", encoding="utf-8")
docs = loader.load()

splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
chunks = splitter.split_documents(docs)
total_chunks = len(chunks)
print(f"Total chunks generated: {total_chunks}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=google_api_key,
)

# Pehle Pinecone connection banao (data direct nahi bhej rahe)
vectorstore = PineconeVectorStore(
    index_name="sjcem-chatbot",
    embedding=embeddings,
    pinecone_api_key=pinecone_api_key,
)

# 80 chunks ka batch size rakha hai taaki 100/min ki limit cross na ho
BATCH_SIZE = 80

for i in range(0, total_chunks, BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    print(f"Uploading batch {i + 1} to {min(i + BATCH_SIZE, total_chunks)} out of {total_chunks}...")
    
    # Batch upload karo
    vectorstore.add_documents(batch)
    
    # Agar aur batches bache hain, toh 60 seconds wait karo limit reset hone ke liye
    if i + BATCH_SIZE < total_chunks:
        print("⏳ Google API limit prevention: Waiting for 60 seconds before next batch...")
        time.sleep(60)

print("✅ Success! Saara data Pinecone Cloud DB me safe hai.")