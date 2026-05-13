import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Paths
DATA_PATH = "data/raw/recipe_chunks.json"
INDEX_PATH = "data/index"  # Directory for FAISS index
os.makedirs(INDEX_PATH, exist_ok=True)

print("🔍 Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("📖 Loading and embedding chunks...")
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"Total chunks loaded: {len(all_chunks)}")
    
    documents = []
    for i, item in enumerate(all_chunks):
        if not isinstance(item, dict) or "text_chunk" not in item or "title" not in item:
            print(f"Skipping invalid chunk at index {i}: {item}")
            continue
        documents.append(Document(page_content=item["text_chunk"], metadata={"title": item["title"]}))

    print(f"Number of valid documents: {len(documents)}")
    if documents:
        print(f"Sample document: {documents[0]}")
    else:
        print("No valid documents were created! Check the JSON file format.")
        exit(1)
except json.JSONDecodeError as e:
    print(f"JSON decoding error: {str(e)}. Please verify the file format of {DATA_PATH}.")
    exit(1)
except Exception as e:
    print(f"Error loading file: {str(e)}")
    exit(1)

# Create FAISS index with LangChain
print("📦 Creating FAISS index...")
try:
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(INDEX_PATH)  # Save using LangChain's method
    print(f"\n✅ Saved FAISS index to: {INDEX_PATH}")
except Exception as e:
    print(f"Error creating FAISS index: {str(e)}")
    exit(1)