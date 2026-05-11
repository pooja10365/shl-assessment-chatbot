import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Lazy globals
model = None
index = None
catalog = None


def load_resources():
    global model, index, catalog

    # Load catalog first (lightweight)
    if catalog is None:
        with open("catalog.json", "r", encoding="utf-8") as f:
            catalog = json.load(f)

    # Load model only when needed
    if model is None:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    # Build FAISS index only once
    if index is None:

        descriptions = [
            item.get("description", "")
            for item in catalog
        ]

        embeddings = model.encode(
            descriptions,
            convert_to_numpy=True
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(
            embeddings.astype("float32")
        )


def search_assessments(query, top_k=5):

    load_resources()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(catalog[idx])

    return results