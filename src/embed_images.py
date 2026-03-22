import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import clip
from tqdm import tqdm

SCENES_INDEX = Path("data/kitti/scenes_index.json")
EMBEDDINGS_DIR = Path("outputs/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    return model, preprocess


def embed_images(batch_size: int = 32):
    with open(SCENES_INDEX) as f:
        scenes = json.load(f)

    model, preprocess = load_model()

    embeddings = {}

    for i in tqdm(range(0, len(scenes), batch_size), desc="Embedding images"):
        batch = scenes[i : i + batch_size]

        images = []
        ids = []
        for s in batch:
            img = Image.open(s["image_path"]).convert("RGB")
            images.append(preprocess(img))
            ids.append(s["scene_id"])

        image_tensor = torch.stack(images).to(DEVICE)

        with torch.no_grad():
            feats = model.encode_image(image_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

        for scene_id, vec in zip(ids, feats.cpu().numpy()):
            embeddings[scene_id] = vec

    # Save as .npy + index
    all_ids = list(embeddings.keys())
    all_vecs = np.stack([embeddings[sid] for sid in all_ids])

    np.save(EMBEDDINGS_DIR / "image_embeddings.npy", all_vecs)
    with open(EMBEDDINGS_DIR / "image_ids.json", "w") as f:
        json.dump(all_ids, f)

    print(f"Saved {len(all_ids)} embeddings, shape: {all_vecs.shape}")
    return all_ids, all_vecs


def sanity_check(all_ids, all_vecs):
    """Cosine similarity between scene_000000 and scene_000001 vs scene_000099."""
    def cos_sim(a, b):
        return float(np.dot(a, b))  # already L2 normalized

    v0 = all_vecs[0]
    v1 = all_vecs[1]
    v99 = all_vecs[min(99, len(all_vecs)-1)]

    print(f"\nSanity check:")
    print(f"  sim(scene_000000, scene_000001) = {cos_sim(v0, v1):.4f}  ← expect high (adjacent frames)")
    print(f"  sim(scene_000000, scene_000099) = {cos_sim(v0, v99):.4f}  ← expect lower (far frames)")


if __name__ == "__main__":
    all_ids, all_vecs = embed_images(batch_size=32)
    sanity_check(all_ids, all_vecs)
