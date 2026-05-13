import os
import json
import pandas as pd
from tqdm import tqdm

# File paths
DATA_PATH = "data/RecipeNLG_dataset.csv"
SAVE_PATH = "data/raw/recipe_chunks.json"
os.makedirs("data/raw", exist_ok=True)

# Load CSV
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Drop rows with missing values
df = df.dropna(subset=["title", "NER", "directions"])

# Optional: limit size for quick testing
# df = df.sample(5000, random_state=42)

def clean_ner(ner_str):
    items = ner_str.split("', '")
    return [item.replace("['", "").replace("']", "").strip() for item in items]

def chunk_recipe(title, ingredients, instructions):
    chunks = []
    base_text = f"Title: {title}\nIngredients: {ingredients}\nDirections: {instructions}"
    words = base_text.split()

    for i in range(0, len(words), 300):
        chunk = " ".join(words[i:i + 300])
        chunks.append({
            "title": title,
            "text_chunk": chunk
        })
    return chunks

# Build chunks
print("🔪 Chunking recipes...")
all_chunks = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    title = row["title"]
    instructions = row["directions"]
    ingredients = ", ".join(clean_ner(row["NER"]))
    recipe_chunks = chunk_recipe(title, ingredients, instructions)
    all_chunks.extend(recipe_chunks)

# Save as JSON
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(all_chunks)} chunks to {SAVE_PATH}")
