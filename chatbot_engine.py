import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_texts_from_data_folder():
    texts, sources = [], []
    for file in ["ra_12009.txt", "irr_ra_12009.txt"]:
        with open(file, "r", encoding="utf-8") as f:
            raw = f.read()
            paragraphs = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 50]
            texts.extend(paragraphs)
            sources.extend([file] * len(paragraphs))
    return texts, sources

texts, sources = load_texts_from_data_folder()
embeddings = model.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def semantic_search(query, top_k=3):
    q_vec = model.encode([query], convert_to_numpy=True)
    _, idx = index.search(q_vec, top_k)
    return [(sources[i], texts[i]) for i in idx[0]]
