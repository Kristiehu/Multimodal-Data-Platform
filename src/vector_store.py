import json
import time
import numpy as np
import faiss
from pathlib import Path

EMBEDDINGS_DIR = Path("outputs/embeddings")
INDEX_DIR      = Path("outputs/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH  = INDEX_DIR / "faiss.index"
ID_MAP_PATH = INDEX_DIR / "id_map.json"


class VectorStore:
    def __init__(self, dim: int):
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on L2-normed vecs
        self.ids   = []                       # position → scene_id

    def add(self, scene_ids: list, vectors: np.ndarray):
        assert vectors.shape[1] == self.dim, f"Expected dim {self.dim}, got {vectors.shape[1]}"
        vecs = vectors.astype(np.float32)
        self.index.add(vecs)
        self.ids.extend(scene_ids)
        print(f"Index now contains {self.index.ntotal} vectors")

    def search(self, query_vec: np.ndarray, k: int = 10):
        q = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"scene_id": self.ids[idx], "score": float(score)})
        return results

    def save(self):
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(ID_MAP_PATH, "w") as f:
            json.dump(self.ids, f)
        print(f"Saved index → {INDEX_PATH}")

    @classmethod
    def load(cls):
        index = faiss.read_index(str(INDEX_PATH))
        with open(ID_MAP_PATH) as f:
            ids = json.load(f)
        store = cls(dim=index.d)
        store.index = index
        store.ids   = ids
        return store


def build_index():
    fused_vecs = np.load(EMBEDDINGS_DIR / "fused_embeddings.npy")
    with open(EMBEDDINGS_DIR / "fused_ids.json") as f:
        fused_ids = json.load(f)

    dim   = fused_vecs.shape[1]
    store = VectorStore(dim=dim)
    store.add(fused_ids, fused_vecs)
    store.save()
    return store


def benchmark(store: VectorStore, n_queries: int = 50):
    print(f"\nBenchmark — {n_queries} random queries:")
    fused_vecs = np.load(EMBEDDINGS_DIR / "fused_embeddings.npy")

    times = []
    for _ in range(n_queries):
        q = fused_vecs[np.random.randint(len(fused_vecs))]
        t0 = time.perf_counter()
        store.search(q, k=5)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"  mean : {np.mean(times):.2f} ms")
    print(f"  p99  : {np.percentile(times, 99):.2f} ms")
    print(f"  max  : {np.max(times):.2f} ms")


if __name__ == "__main__":
    store = build_index()

    # Self-retrieval sanity check — query with scene_000000, expect it as top result
    fused_vecs = np.load(EMBEDDINGS_DIR / "fused_embeddings.npy")
    with open(EMBEDDINGS_DIR / "fused_ids.json") as f:
        fused_ids = json.load(f)

    q = fused_vecs[0]
    results = store.search(q, k=5)
    print(f"\nSelf-retrieval check (query = {fused_ids[0]}):")
    for r in results:
        print(f"  {r['scene_id']}  score={r['score']:.4f}")

    benchmark(store)
