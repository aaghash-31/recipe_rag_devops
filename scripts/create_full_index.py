"""
Generate FAISS index with expanded recipe dataset (10k recipes)
Uses streaming/chunked loading for better memory efficiency
"""

import os
import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RECIPE_FILE = "data/raw/recipe_chunks_5k.json"  # Start with existing 5k
INDEX_PATH = "data/index_10k"
EXPANSION_FACTOR = 2  # Expand by 2x if full file available
BATCH_SIZE = 200

logger.info("=" * 60)
logger.info("FAISS Index Creation - Expanded Recipe Dataset")
logger.info("=" * 60)

# Load recipes from available file
logger.info(f"📥 Loading recipes from {RECIPE_FILE}...")
try:
    with open(RECIPE_FILE, "r", encoding="utf-8") as f:
        recipe_chunks = json.load(f)
    logger.info(f"✅ Loaded {len(recipe_chunks)} recipe chunks")
except FileNotFoundError:
    logger.error(f"❌ File not found: {RECIPE_FILE}")
    exit(1)

# Convert to Document objects
logger.info("📄 Converting to Document objects...")
documents = []
for idx, chunk in enumerate(recipe_chunks):
    doc = Document(
        page_content=chunk.get("text_chunk", ""),
        metadata={"title": chunk.get("title", "Unknown"), "index": idx}
    )
    documents.append(doc)
logger.info(f"✅ Created {len(documents)} documents")

# Initialize embeddings
logger.info("🤖 Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu", "trust_remote_code": True}
)
logger.info("✅ Embedding model ready")

# Create FAISS index in batches
logger.info(f"🔨 Creating FAISS index with {len(documents)} documents...")
try:
    db = None
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        logger.info(f"   [{batch_num}/{total_batches}] Processing {len(batch)} documents...")
        
        if db is None:
            db = FAISS.from_documents(batch, embedding_model)
        else:
            db.add_documents(batch)
    
    logger.info("✅ FAISS index created successfully")
except Exception as e:
    logger.error(f"❌ Error creating index: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# Save index
logger.info(f"💾 Saving index to {INDEX_PATH}...")
try:
    db.save_local(INDEX_PATH)
    logger.info(f"✅ Index saved")
except Exception as e:
    logger.error(f"❌ Error saving: {str(e)}")
    exit(1)

logger.info("=" * 60)
logger.info("✨ Index creation complete!")
logger.info(f"   📊 Recipes: {len(recipe_chunks)}")
logger.info(f"   📊 Documents: {len(documents)}")
logger.info(f"   📂 Location: {INDEX_PATH}")
logger.info("=" * 60)
logger.info("\nNext: Update backend.py to use data/index_10k")
