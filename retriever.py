from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss

# Lazy globals
model = None
index = None
catalog = None


def load_resources():
    global model, index, catalog

    if model is None:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    if catalog is None:
        with open("catalog.json", "r", encoding="utf-8") as f:
            catalog = json.load(f)

    if index is None:
        embeddings = model.encode(
            [item["description"] for item in catalog]
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))


def search_assessments(query, top_k=5):

    load_resources()

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(catalog[idx])

    return results
