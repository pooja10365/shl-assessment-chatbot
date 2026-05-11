import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# -----------------------------
# Load Catalog
# -----------------------------
with open("catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)


# -----------------------------
# Load Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Prepare Search Text
# -----------------------------
documents = []

for item in catalog:

    text = (
        item.get("name", "") + " " +
        item.get("description", "") + " " +
        " ".join(item.get("keys", []))
    )

    documents.append(text)


# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = model.encode(documents)


# Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")


# -----------------------------
# Build FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# -----------------------------
# Search Function
# -----------------------------
def search_assessments(query, top_k=5):

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:
        results.append(catalog[idx])

    return results