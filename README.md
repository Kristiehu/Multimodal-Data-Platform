# Multimodal LiDAR Scene Retrieval

Text-to-scene search over KITTI driving data using CLIP + FAISS.
Type a natural-language description → get back the most visually and geometrically matching scenes.

---

## How it works

Each scene is represented by a fused embedding:

```
fused = normalise(0.7 × CLIP(RGB image) + 0.3 × CLIP(LiDAR BEV image))
```

Text queries are encoded with the same CLIP text encoder (512-d) and searched via FAISS cosine similarity. A fine-tuned checkpoint (`outputs/clip_finetuned.pt`) is used automatically if present.

---

## Setup

```bash
pip install -r requirements.txt
```

Data: place the KITTI 3D Object Detection dataset under `data/kitti_raw/`.

---

## Usage

**1. Build the index (one-time, ~30–60 min for all 7481 scenes)**
```bash
PYTHONPATH=src python src/ingest_pipeline.py
```

**2. Generate evaluation pairs**
```bash
PYTHONPATH=src python src/generate_eval_pairs.py
```

**3. Fine-tune CLIP (optional but recommended)**
```bash
PYTHONPATH=src python src/finetune_clip.py
```

**4. Launch the app**
```bash
PYTHONPATH=src streamlit run src/app.py
```

---

## Project structure

```
src/
  app.py                  Streamlit UI
  ingest_pipeline.py      Full data → index pipeline
  data_loader.py          KITTI scenes → disk (up to 7481 scenes)
  caption_generator.py    Rich spatial captions from 3D labels
  embed_images.py         CLIP image embeddings (512-d)
  bev_projection.py       LiDAR → BEV image → embedding, fused with image
  vector_store.py         FAISS index (IndexFlatIP)
  retrieval.py            Text query → top-k scenes + Recall@k eval
  finetune_clip.py        CLIP fine-tuning with multi-variant captions
  generate_eval_pairs.py  Semantic ground-truth pairs for evaluation

data/
  kitti_raw/              Raw KITTI files (velodyne, image_2, label_2)
  kitti/scenes/           Processed scenes (image.jpg, pointcloud.pcd, meta.json)

outputs/
  embeddings/             Saved .npy embeddings + FAISS index
  bev_images/             BEV top-down renders
  clip_finetuned.pt       Fine-tuned checkpoint (if trained)
  eval_pairs.json         Semantic evaluation ground truth
```

---

## Evaluation

```bash
PYTHONPATH=src python src/retrieval.py
```

Recall@5 is measured against `outputs/eval_pairs.json` — semantically grounded (query, scene_id) pairs across 8 categories: many cars, pedestrians, cyclists, trucks, empty road, urban mixed, close objects, distant objects.
