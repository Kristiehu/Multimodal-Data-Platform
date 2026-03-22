"""
bev_projection.py
-----------------
LiDAR point-cloud → Bird's-Eye-View (BEV) image → CLIP embedding.

Fusion change (v2)
------------------
Previously: img(512) ⊕ bev(512) → PCA(200-d) for the index, and
            text(512) duplicated → PCA(200-d) for queries — causing
            a severe modality gap.

Now: fused(512) = normalise( 0.7 × img_embed + 0.3 × bev_embed )
  • Image component dominates (0.7) so text ↔ image CLIP alignment is preserved.
  • BEV adds geometric structure (0.3) as a soft geometric prior.
  • Text queries use the native CLIP 512-d space — perfect alignment.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d
import torch
import clip
from tqdm import tqdm

SCENES_INDEX   = Path("data/kitti/scenes_index.json")
EMBEDDINGS_DIR = Path("outputs/embeddings")
BEV_DIR        = Path("outputs/bev_images")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
BEV_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BEV config — KITTI forward = +X, left = +Y, up = +Z (Velodyne frame)
X_RANGE = (0, 50)       # metres forward
Y_RANGE = (-25, 25)     # metres left/right
Z_RANGE = (-3, 1)       # metres height
BEV_SIZE = (256, 256)   # output image pixels

# Fusion weights — keep image dominant so text↔image alignment is preserved
IMG_WEIGHT = 0.70
BEV_WEIGHT = 0.30


def pcd_to_bev(pcd_path: str) -> Image.Image:
    """Project a LiDAR point cloud (.pcd) to a coloured BEV image."""
    points = np.asarray(o3d.io.read_point_cloud(pcd_path).points)

    mask = (
        (points[:, 0] >= X_RANGE[0]) & (points[:, 0] <= X_RANGE[1]) &
        (points[:, 1] >= Y_RANGE[0]) & (points[:, 1] <= Y_RANGE[1]) &
        (points[:, 2] >= Z_RANGE[0]) & (points[:, 2] <= Z_RANGE[1])
    )
    points = points[mask]

    if len(points) == 0:
        return Image.fromarray(np.zeros((*BEV_SIZE, 3), dtype=np.uint8))

    px = ((points[:, 0] - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])
          * (BEV_SIZE[0] - 1)).astype(int)
    py = ((points[:, 1] - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
          * (BEV_SIZE[1] - 1)).astype(int)
    z_norm = ((points[:, 2] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
              * 255).astype(np.uint8)

    canvas = np.zeros((*BEV_SIZE, 3), dtype=np.uint8)
    canvas[px, py, 0] = z_norm          # R = height
    canvas[px, py, 1] = 128             # G = constant (visibility)
    canvas[px, py, 2] = 255 - z_norm    # B = inverse height

    return Image.fromarray(canvas)


def embed_bev(batch_size: int = 32):
    """Encode all BEV images with CLIP and save bev_embeddings.npy."""
    with open(SCENES_INDEX) as f:
        scenes = json.load(f)

    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

    bev_embeddings: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(scenes), batch_size), desc="BEV embedding"):
        batch = scenes[i : i + batch_size]
        bevs, ids = [], []

        for s in batch:
            bev_img  = pcd_to_bev(s["pcd_path"])
            bev_path = BEV_DIR / f"{s['scene_id']}_bev.jpg"
            bev_img.save(bev_path)
            bevs.append(preprocess(bev_img))
            ids.append(s["scene_id"])

        tensor = torch.stack(bevs).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        for scene_id, vec in zip(ids, feats.cpu().numpy()):
            bev_embeddings[scene_id] = vec

    all_ids  = list(bev_embeddings.keys())
    bev_vecs = np.stack([bev_embeddings[sid] for sid in all_ids])

    np.save(EMBEDDINGS_DIR / "bev_embeddings.npy", bev_vecs)
    print(f"Saved BEV embeddings: {bev_vecs.shape}")
    return all_ids, bev_vecs


def fuse_embeddings():
    """
    Fuse image + BEV embeddings via weighted average → 512-d normalised vector.

    Replaces the old PCA-based concatenation which required a broken
    text-duplication trick at query time.  The resulting fused vector sits
    in the same 512-d CLIP space as text queries.
    """
    img_vecs = np.load(EMBEDDINGS_DIR / "image_embeddings.npy")   # (N, 512)
    bev_vecs = np.load(EMBEDDINGS_DIR / "bev_embeddings.npy")     # (N, 512)

    with open(EMBEDDINGS_DIR / "image_ids.json") as f:
        ids = json.load(f)

    assert img_vecs.shape == bev_vecs.shape, (
        f"Shape mismatch: img {img_vecs.shape} vs bev {bev_vecs.shape}. "
        "Re-run embed_images() and embed_bev() for the same scenes list."
    )

    # Weighted average → re-normalise to unit sphere
    fused = IMG_WEIGHT * img_vecs + BEV_WEIGHT * bev_vecs          # (N, 512)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    fused = fused / np.clip(norms, 1e-8, None)

    np.save(EMBEDDINGS_DIR / "fused_embeddings.npy", fused)
    with open(EMBEDDINGS_DIR / "fused_ids.json", "w") as f:
        json.dump(ids, f)

    # Quick sanity: average cosine sim between consecutive scenes
    cos_sim_01 = float(np.dot(fused[0], fused[1]))
    print(f"Fused embeddings: {fused.shape}")
    print(f"  img_weight={IMG_WEIGHT}  bev_weight={BEV_WEIGHT}")
    print(f"  Norm check (should be ~1.0): {np.linalg.norm(fused[0]):.4f}")
    print(f"  Cosine sim(scene_0, scene_1): {cos_sim_01:.4f}")
    return ids, fused


if __name__ == "__main__":
    all_ids, bev_vecs = embed_bev(batch_size=32)
    ids, fused = fuse_embeddings()
    print(f"\nBEV + fusion complete. Fused shape: {fused.shape}")
