"""
ingest_pipeline.py
------------------
Full ingestion pipeline: raw KITTI → embeddings → FAISS index.

Run (from project root):
    PYTHONPATH=src python src/ingest_pipeline.py

Pipeline steps
--------------
  1. Build scene dataset   (data_loader)     – up to 7481 scenes with captions
  2. Image embeddings      (embed_images)    – CLIP ViT-B/32, 512-d
  3. BEV embeddings        (bev_projection)  – CLIP on LiDAR BEV images, 512-d
  4. Fuse embeddings       (bev_projection)  – weighted average → 512-d
  5. Build FAISS index     (vector_store)    – IndexFlatIP
  6. Generate eval pairs   (generate_eval_pairs) – semantic ground truth

"""
import sys
import time
sys.path.insert(0, "src")

from data_loader      import build_scene_dataset
from embed_images     import embed_images
from bev_projection   import embed_bev, fuse_embeddings
from vector_store     import build_index


def run(max_scenes: int = 7481):
    t0 = time.time()

    print("=" * 55)
    print(f"STEP 1/5  Build scene dataset  (max {max_scenes} scenes)")
    print("=" * 55)
    build_scene_dataset(max_scenes=max_scenes)

    print("\n" + "=" * 55)
    print("STEP 2/5  Image embeddings (CLIP ViT-B/32)")
    print("=" * 55)
    embed_images(batch_size=32)

    print("\n" + "=" * 55)
    print("STEP 3/5  BEV projection + LiDAR embeddings")
    print("=" * 55)
    embed_bev(batch_size=32)

    print("\n" + "=" * 55)
    print("STEP 4/5  Fuse embeddings (weighted average → 512-d)")
    print("=" * 55)
    fuse_embeddings()

    print("\n" + "=" * 55)
    print("STEP 5/5  Build FAISS index")
    print("=" * 55)
    build_index()

    elapsed = time.time() - t0
    print(f"\n{'=' * 55}")
    print(f"Pipeline complete in {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"{'=' * 55}")



if __name__ == "__main__":
    # Pass max_scenes as CLI arg: python src/ingest_pipeline.py 500
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7481
    run(max_scenes=n)
