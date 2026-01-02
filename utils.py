import os
import re
import json
import hashlib
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# CONFIG
INDEX_DIR = "store"
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
DIM = 384

INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.json")

# HELPERS
def normalize_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def hash_text(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()

# PDF 
def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)

# STORE 
def load_store():
    """
    Safely load FAISS index and metadata.
    If either file is missing, initialize a clean store.
    """

    # Ensure store directory exists
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Case 1: both index and metadata exist
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            return index, metadata
        except Exception:
            pass 

    # Case 2: fresh initialization
    index = faiss.IndexFlatIP(DIM)
    metadata = []

    # Persist empty store
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return index, metadata

def save_store(index, metadata):
    """
    Persist FAISS index and metadata safely to disk.
    """

    os.makedirs(INDEX_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# ADD DOCUMENT
def add_document(text: str, source: str):
    index, metadata = load_store()

    sentences = [
        normalize_text(s)
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 40
    ]

    existing_hashes = {m["hash"] for m in metadata}

    new_texts = []
    new_meta = []

    for s in sentences:
        h = hash_text(s)
        if h not in existing_hashes:
            new_texts.append(s)
            new_meta.append({
                "text": s,
                "source": source,
                "hash": h
            })

    if not new_texts:
        return

    vecs = EMBED_MODEL.encode(new_texts, normalize_embeddings=True)
    index.add(np.array(vecs).astype("float32"))
    metadata.extend(new_meta)

    save_store(index, metadata)

# QUERY 
def retrieve(query: str, top_k=6) -> List[Dict]:
    index, metadata = load_store()
    if index.ntotal == 0:
        return []

    qvec = EMBED_MODEL.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(qvec).astype("float32"), top_k)

    results = []
    for i, score in zip(ids[0], scores[0]):
        if i < len(metadata):
            results.append({
                "score": float(score),
                **metadata[i]
            })

    return results

# ANSWER ENGINE 
def synthesize_answer(query: str, hits: List[Dict]):
    if not hits:
        return "No relevant information was found in the uploaded documents.", []

    # Detect list intent
    is_list = any(k in query.lower() for k in ["list", "laws", "acts", "units", "types"])

    # Clean evidence
    evidence = []
    seen = set()

    for h in hits:
        t = h["text"]
        key = t.lower()
        if key not in seen:
            seen.add(key)
            evidence.append(t)

    if is_list:
        items = []
        for t in evidence:
            if len(t) < 200:
                items.append(t)

        if items:
            answer = "Based on the retrieved documents, the following items are identified:\n\n"
            for i, it in enumerate(items, 1):
                answer += f"{i}. {it}\n"
            return answer.strip(), hits

    # Narrative synthesis
    answer = " ".join(evidence[:4])
    return answer, hits
