import json
import numpy as np
import torch
import clip
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.decomposition import PCA
from vector_store import VectorStore

EMBEDDINGS_DIR = Path("outputs/embeddings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Multimodal LiDAR Retrieval API", version="1.0.0")

# Global state — loaded once at startup
_model    = None
_pca      = None
_store    = None


@app.on_event("startup")
def load_models():
    global _model, _pca, _store
    print("Loading CLIP model...")
    _model, _ = clip.load("ViT-B/32", device=DEVICE)
    _model.eval()

    print("Fitting PCA...")
    img_vecs = np.load(EMBEDDINGS_DIR / "image_embeddings.npy")
    bev_vecs = np.load(EMBEDDINGS_DIR / "bev_embeddings.npy")
    combined = np.concatenate([img_vecs, bev_vecs], axis=1)
    _pca = PCA(n_components=200, random_state=42)
    _pca.fit(combined)

    print("Loading FAISS index...")
    _store = VectorStore.load()

    print(f"Ready — {_store.index.ntotal} scenes indexed")


def encode_query(query: str) -> np.ndarray:
    tokens = clip.tokenize([query]).to(DEVICE)
    with torch.no_grad():
        feat = _model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vec      = feat.cpu().numpy().astype(np.float32)
    combined = np.concatenate([vec, vec], axis=1)
    proj     = _pca.transform(combined)
    proj     = proj / np.linalg.norm(proj, axis=1, keepdims=True)
    return proj[0]


# ---------- schemas ----------

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResult(BaseModel):
    scene_id: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.get("/stats")
def stats():
    return {
        "total_scenes": _store.index.ntotal,
        "embedding_dim": _store.dim,
        "index_type": "IndexFlatIP",
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.k < 1 or req.k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")

    q_vec   = encode_query(req.query)
    results = _store.search(q_vec, k=req.k)

    return SearchResponse(
        query=req.query,
        results=[SearchResult(**r) for r in results]
    )
