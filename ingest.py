from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db():
    print("Loading data...")
    # 1. Load the Markdown file
    loader = TextLoader("college_data_cleaned.md", encoding="utf-8")
    docs = loader.load()

    # 2. Split into chunks
    splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Use Local Open-Source Embeddings (Fast, Free, No API Key needed)
    print("Loading local embedding model (Pehli baar thoda time lega download hone me)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Save to ChromaDB
    print("Saving to ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print("✅ Data successfully saved to local ChromaDB!")

if __name__ == "__main__":
    build_vector_db()