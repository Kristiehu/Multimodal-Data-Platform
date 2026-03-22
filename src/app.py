"""
app.py — Multimodal LiDAR Scene Retrieval (Streamlit UI)

Architecture (v2)
-----------------
Text query → CLIP text encoder (512-d) → FAISS IndexFlatIP
Scene index built from:  normalise(0.7 × img_embed + 0.3 × bev_embed)
Fine-tuned checkpoint loaded automatically if present.

The previous PCA-duplication trick has been removed; text and scene vectors
now live in the same native CLIP 512-d embedding space.
"""

import sys
sys.path.insert(0, "src")

import json
import numpy as np
import torch
import clip
from pathlib import Path
from PIL import Image
import streamlit as st
from vector_store import VectorStore

EMBEDDINGS_DIR = Path("outputs/embeddings")
BEV_DIR        = Path("outputs/bev_images")
CKPT_PATH      = Path("outputs/clip_finetuned.pt")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Multimodal LiDAR Retrieval",
    page_icon="🚗",
    layout="wide",
)


# ── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    model, _ = clip.load("ViT-B/32", device=DEVICE)
    if CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.eval()
    store = VectorStore.load()
    return model, store


def encode_query(query: str, model) -> np.ndarray:
    """
    Encode a text query into the native CLIP 512-d embedding space.
    No PCA, no duplication — text and scene vectors are directly comparable.
    """
    tokens = clip.tokenize([query], truncate=True).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)[0]


# ── UI ─────────────────────────────────────────────────────────────────────

st.title("🚗 Multimodal LiDAR Scene Retrieval")
st.caption("CLIP ViT-B/32 · FAISS · KITTI · Fused Image + BEV embeddings")

with st.sidebar:
    st.header("About")
    st.markdown("""
**Stack**
- CLIP ViT-B/32 (fine-tuned on KITTI)
- FAISS IndexFlatIP (cosine similarity)
- BEV projection (Open3D · LiDAR → top-down image)
- Up to 7 481 KITTI training scenes

**Embedding space (v2)**
1. RGB image → CLIP image encoder → 512-d
2. LiDAR BEV image → CLIP image encoder → 512-d
3. Fused = normalise(0.7 × img + 0.3 × bev) → **512-d**
4. Text query → CLIP text encoder → **512-d** (native alignment)

*No PCA projection or duplication hack — text and scene vectors
live in the same CLIP embedding space.*

**Fine-tuning**
- Rich spatial captions (near/far, left/right, scene type)
- Up to 3 caption variants per scene (data augmentation)
- Last 2 visual transformer blocks + full text encoder unfrozen
    """)
    k = st.slider("Results to show (k)", 1, 10, 5)

# ── Load & status ─────────────────────────────────────────────────────────────

try:
    model, store = load_resources()
    ft_label = " · fine-tuned ✓" if CKPT_PATH.exists() else " · base CLIP"
    st.success(f"Index loaded — {store.index.ntotal} scenes ready{ft_label}", icon="✅")
except Exception as e:
    st.error(f"Failed to load index: {e}\n\nRun the ingest pipeline first:\n"
             "`PYTHONPATH=src python src/ingest_pipeline.py`")
    st.stop()

# ── Example queries ────────────────────────────────────────────────────────────

examples = [
    "a car driving closely in front",
    "pedestrians crossing the road",
    "empty road with no vehicles",
    "busy urban intersection with cars",
    "a cyclist on the right side",
    "truck on a highway",
]

st.markdown("**Try an example:**")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input(
    "Search scenes by description",
    value=st.session_state.get("query", ""),
    placeholder="e.g. a road with parked cars near buildings",
)

# ── Search & display ──────────────────────────────────────────────────────────

if query:
    with st.spinner("Searching…"):
        q_vec   = encode_query(query, model)
        results = store.search(q_vec, k=k)

    st.markdown(f"### Top {k} results for: *{query}*")
    st.divider()

    for r in results:
        scene_id  = r["scene_id"]
        score     = r["score"]
        scene_dir = Path("data/kitti/scenes") / scene_id

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            img_path = scene_dir / "image.jpg"
            if img_path.exists():
                st.image(str(img_path), caption=f"RGB — {scene_id}", use_column_width=True)
            else:
                st.warning(f"Image not found: {img_path}")

        with col2:
            bev_path = BEV_DIR / f"{scene_id}_bev.jpg"
            if bev_path.exists():
                st.image(str(bev_path), caption="BEV (LiDAR top-down)", use_column_width=True)
            else:
                st.info("BEV image not generated yet.")

        with col3:
            st.metric("Cosine Score", f"{score:.4f}")
            meta_path = scene_dir / "meta.json"
            if meta_path.exists():
                meta    = json.load(open(meta_path))
                caption = meta.get("caption", "—")
                st.caption(f"**Caption:** {caption}")
                st.caption(f"Source: {meta.get('source', 'kitti')}")

        st.divider()
