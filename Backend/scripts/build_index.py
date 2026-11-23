import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from ingredients.normalizer import normalize_ingredient_list
from ingredients.ingredient_index import IngredientIndex
DATA_PATH = os.path.join(BASE_DIR, "data", "Recipedb_kk.csv")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "data", "recipes_faiss.index")
METADATA_PATH = os.path.join(BASE_DIR, "data", "recipes_metadata.pkl")
INGREDIENT_INDEX_PATH = os.path.join(BASE_DIR, "data", "ingredient_index.pkl")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128   

def build_index_main():
    # ================= LOAD CSV =================
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # ================= SELECT + CLEAN =================
    df = df[
        ["Recipe_ID", "Recipe Name", "Ingredients", "Directions"]
    ].fillna("")

    # ================= RENAME =================
    df = df.rename(columns={
        "Recipe_ID": "recipe_id",
        "Recipe Name": "title",
        "Ingredients": "ingredients",
        "Directions": "instructions"
    })

    def clean_text(text):
        return str(text).replace("[", "").replace("]", "").replace("'", "")

    df["ingredients"] = df["ingredients"].apply(clean_text)
    df["instructions"] = df["instructions"].apply(clean_text)

    print("Normalizing ingredients...")
    df["normalized_ingredients"] = df["ingredients"].apply(
        lambda ing_str: normalize_ingredient_list([i.strip() for i in ing_str.split(",") if i.strip()])
    )

    # ================= BUILD DOCUMENT =================
    print("Building documents...")
    df["document"] = (
        "Recipe Name: " + df["title"] + "\n\n"
        "Ingredients: " + df["ingredients"] + "\n\n"
        "Instructions: " + df["instructions"]
    )

    df = df.reset_index(drop=True)
    print(f"✅ Total recipes indexed: {len(df)}")

    # ================= BUILD INGREDIENT INDEX =================
    print("Building structured ingredient index...")
    recipes_list = df.to_dict("records")
    ing_index = IngredientIndex()
    ing_index.build_from_recipes(recipes_list)
    ing_index.save(INGREDIENT_INDEX_PATH)
    print(f"✅ Ingredient index saved to: {INGREDIENT_INDEX_PATH}")

    # ================= LOAD EMBEDDING MODEL =================
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model onto {device.upper()}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    curr_batch_size = 256 if device == "cuda" else BATCH_SIZE

    embedding_dim = embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(embedding_dim)

    # ================= BUILD FAISS INDEX =================
    print(f"Creating embeddings and building FAISS index on {device.upper()} (batch_size={curr_batch_size})...")
    for start in tqdm(range(0, len(df), curr_batch_size)):
        texts = df["document"].iloc[start:start + curr_batch_size].tolist()
        embeddings = embed_model.encode(
            texts,
            batch_size=curr_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        index.add(np.array(embeddings).astype(np.float32))

    # ================= SAVE =================
    print("Saving FAISS index...")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saving metadata...")
    df[["recipe_id", "title", "ingredients", "normalized_ingredients", "instructions", "document"]].to_pickle(METADATA_PATH)

    print("\n✅ Indexing completed successfully!")

if __name__ == "__main__":
    build_index_main()