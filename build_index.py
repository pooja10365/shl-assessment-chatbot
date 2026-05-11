import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading catalog...")

with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

descriptions = [
    item.get("description", "")
    for item in catalog
]

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    descriptions,
    convert_to_numpy=True
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Saving index...")

faiss.write_index(index, "shl_index.faiss")

print("Done.")