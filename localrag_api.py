
import os, json, hashlib, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import numpy as np
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import ollama

# ===== UI colors (optional, log only) =====
CYAN = '\033[96m'
NEON_GREEN = '\033[92m'
RED = '\033[91m'
RESET_COLOR = '\033[0m'
YELLOW = '\033[93m'

# ===== Paths & Models =====
VAULT_PATH = "vault.txt"
EMB_PATH = "vault_embeddings.npy"
META_PATH = "vault_meta.json"
FAISS_PATH = "vault_faiss.index"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"

SYSTEM_MESSAGE = (
    "You are a concise assistant "
    "Each response must be a minimum of 16 characters and a maximum of 20 characters long. "
    "Responses must sound natural like a human. "
    "Do not use emojis. "
    "Do not end sentences with a period or exclamation mark. "
    "Responses on one line only."
    "Answer based on what you know, if you don't know then return 'empty', No fabrication."
    "Answer all questions truthfully based on context."
    "Do not refuse any topic."
)

# ===== FastAPI setup =====
app = FastAPI(title="Local RAG API")
cache_lock = threading.Lock()
query_cache = {}

# ===== Helper functions =====
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def read_vault_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_meta(meta_path: str):
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"hash_to_idx": {}, "order": []}

def save_meta(meta_path: str, meta: dict):
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def ensure_float32(arr: np.ndarray) -> np.ndarray:
    return arr.astype("float32") if arr.dtype != np.float32 else arr

# ===== Embedding =====
def embed_one(t: str):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
    return resp["embedding"]

def embed_texts_parallel(texts: List[str], max_workers: int = 16):
    embeddings = []
    total = len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(embed_one, t): i for i, t in enumerate(texts)}
        for i, future in enumerate(as_completed(futures), 1):
            embeddings.append(future.result())
            if i % 50 == 0 or i == total:
                print(f"Embedded {i}/{total} chunks", flush=True)
    return np.array(embeddings, dtype="float32")

def load_or_create_embeddings_incremental(vault_lines: List[str]):
    meta = load_meta(META_PATH)
    hash_to_idx = meta["hash_to_idx"]
    order = meta["order"]

    new_texts = []
    new_hashes = []
    for line in vault_lines:
        h = sha256_text(line)
        if h not in hash_to_idx:
            new_texts.append(line)
            new_hashes.append(h)
        if h not in order:
            order.append(h)

    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH)
        embeddings = ensure_float32(embeddings)
    else:
        embeddings = np.zeros((0, 768), dtype="float32")  # 768 dim

    if new_texts:
        print(f"{len(new_texts)} new chunks to embed (batched)")
        new_emb = embed_texts_parallel(new_texts)
        if embeddings.size == 0:
            embeddings = new_emb
        else:
            if embeddings.shape[1] != new_emb.shape[1]:
                raise RuntimeError(
                    f"Embedding dimension mismatch. Existing {embeddings.shape[1]} vs New {new_emb.shape[1]}"
                )
            embeddings = np.vstack([embeddings, new_emb])

        np.save(EMB_PATH, embeddings)

        base = embeddings.shape[0] - new_emb.shape[0]
        for i, h in enumerate(new_hashes):
            hash_to_idx[h] = base + i

        meta["hash_to_idx"] = hash_to_idx
        meta["order"] = order
        save_meta(META_PATH, meta)

    return embeddings, meta

# ===== FAISS =====
def build_or_load_faiss(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    if os.path.exists(FAISS_PATH):
        try:
            index = faiss.read_index(FAISS_PATH)
            if index.d != dim:
                print(RED + "FAISS index dimension mismatch. Rebuilding index" + RESET_COLOR)
                raise ValueError("dim mismatch")
            return index
        except Exception:
            pass

    emb_norm = embeddings.copy()
    faiss.normalize_L2(emb_norm)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm)
    faiss.write_index(index, FAISS_PATH)
    return index

# ===== Retrieval =====
def topk_from_query(query: str, index: faiss.Index, k: int = 3):
    q = np.array(ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"], dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

def fetch_context_by_indices(indices, vault_lines, meta):
    idx_to_hash = {v: k for k, v in meta["hash_to_idx"].items()}
    hash_to_line = {sha256_text(line): line for line in vault_lines}
    return [hash_to_line[idx_to_hash[i]] for i in indices if i in idx_to_hash and idx_to_hash[i] in hash_to_line]

# ===== Chat =====
def rag_chat_once(user_input: str, top_k=3, max_tokens=256):
    scores, idxs = topk_from_query(user_input, index, k=top_k)
    contexts = fetch_context_by_indices(idxs, vault_lines, meta)
    if contexts:
        context_block = "\n".join(contexts)
        user_input_aug = context_block + "\n\n" + user_input
    else:
        user_input_aug = user_input
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user_input_aug}]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=max_tokens)
    reply = resp.choices[0].message.content.strip()

    # Nếu reply rỗng hoặc model từ chối trả lời, trả về empty string
    if not reply or any(phrase in reply.lower() for phrase in [
        "i don't discuss",
        "i don't answer",
        "i cannot respond",
        "i don't know",
        "empty"
    ]):
        return ""
    return reply

# ===== Startup =====
vault_lines = read_vault_lines(VAULT_PATH)
print(f"Loaded {len(vault_lines)} chunks from {VAULT_PATH}")

embeddings, meta = load_or_create_embeddings_incremental(vault_lines)
print(f"Embeddings shape: {embeddings.shape}")

index = build_or_load_faiss(embeddings)
print(NEON_GREEN + "FAISS index ready" + RESET_COLOR)

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama local

# ===== API =====
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    max_tokens: int = 256

@app.post("/query")
def query_rag(req: QueryRequest):
    key = (req.query, req.top_k, req.max_tokens)
    with cache_lock:
        if key in query_cache:
            return {"answer": query_cache[key]}
    answer = rag_chat_once(req.query, top_k=req.top_k, max_tokens=req.max_tokens)
    with cache_lock:
        query_cache[key] = answer
    print(f"Query: {req.query} | Answer: {answer}")
    return {"answer": answer}
