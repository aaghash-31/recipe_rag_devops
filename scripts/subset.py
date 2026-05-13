import ijson
import json

input_path = "data/raw/recipe_chunks.json"
output_path = "data/raw/recipe_chunks_5k.json"

subset = []
count = 0
MAX_RECIPES = 5000

with open(input_path, "r", encoding="utf-8") as f:
    parser = ijson.items(f, "item")  # "item" corresponds to each dict in the top-level list

    for recipe in parser:
        subset.append(recipe)
        count += 1
        if count >= MAX_RECIPES:
            break

with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(subset, out_f, indent=2)

print(f"✅ Saved first {MAX_RECIPES} recipes to {output_path}")
