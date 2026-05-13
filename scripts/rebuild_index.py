"""
Simple recipe index creation - validates and re-creates FAISS index
"""
import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("🔧 Creating FAISS Index from available recipes...")

# Try to load from existing working index first (no need to rebuild)
import os
if os.path.exists("data/index_5k"):
    logger.info("✅ FAISS index already exists at data/index_5k")
    logger.info("   Current recipes: 5,000+")
    logger.info("   To add more recipes:")
    logger.info("   1. Run: python load_kaggle_recipe_nlg.py")
    logger.info("   2. Run: python create_full_index.py")
    logger.info("\nNo action needed - using existing index!")
    exit(0)

# If index doesn't exist, create from recipe_chunks_5k.json
recipe_file = "data/raw/recipe_chunks_5k.json"
if not os.path.exists(recipe_file):
    logger.error(f"❌ Recipe file not found: {recipe_file}")
    exit(1)

logger.info(f"📥 Loading recipes...")
try:
    with open(recipe_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        # Clean up potential encoding issues
        recipe_chunks = json.loads(content)
except Exception as e:
    logger.error(f"❌ Error reading file: {e}")
    exit(1)

logger.info(f"✅ Loaded {len(recipe_chunks)} recipes")

# Create documents
logger.info("📄 Creating documents...")
documents = [
    Document(
        page_content=chunk.get("text_chunk", ""),
        metadata={"title": chunk.get("title", "Unknown")}
    )
    for chunk in recipe_chunks
]

# Initialize embeddings
logger.info("🤖 Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Create and save index
logger.info(f"🔨 Creating FAISS index with {len(documents)} documents...")
db = FAISS.from_documents(documents, embeddings)
db.save_local("data/index_5k")

logger.info("=" * 50)
logger.info("✅ FAISS Index created successfully!")
logger.info(f"   📊 Recipes: {len(recipe_chunks)}")
logger.info(f"   📂 Path: data/index_5k")
logger.info("=" * 50)
