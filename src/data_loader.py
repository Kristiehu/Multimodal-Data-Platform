"""
data_loader.py
--------------
Load KITTI raw data into a consistent scene dataset.

Key change from v1: max_scenes now defaults to ALL available frames (7481),
and each scene's meta.json is enriched with a primary CLIP caption derived
from the label file so downstream modules don't have to re-parse labels.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import open3d as o3d

RAW_BASE   = Path("data/kitti_raw/kitti-3d-object-detection-dataset/training")
LIDAR_DIR  = RAW_BASE / "velodyne"
IMAGE_DIR  = RAW_BASE / "image_2"
LABEL_DIR  = RAW_BASE / "label_2"
SCENES_DIR = Path("data/kitti/scenes")
SCENES_DIR.mkdir(parents=True, exist_ok=True)


def bin_to_pcd(bin_path: Path, out_path: Path):
    """Convert KITTI Velodyne .bin to Open3D .pcd (XYZ only)."""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(str(out_path), pcd)


def find_kitti_pairs(max_scenes: int = 7481) -> list[tuple]:
    """
    Return a list of (bin_path, img_path, label_path, scene_id) for every
    frame that has a matching image file, up to max_scenes.
    """
    pairs = []
    bin_files = sorted(LIDAR_DIR.glob("*.bin"))[:max_scenes]
    for bin_path in bin_files:
        img_path   = IMAGE_DIR / (bin_path.stem + ".png")
        label_path = LABEL_DIR / (bin_path.stem + ".txt")
        if img_path.exists():
            pairs.append((bin_path, img_path, label_path, f"scene_{bin_path.stem}"))
    return pairs


def build_scene_dataset(max_scenes: int = 7481) -> list[dict]:
    """
    Build / refresh the scene dataset on disk.

    For each frame:
      - Convert .bin → .pcd  (skipped if already done)
      - Copy .png → .jpg       (skipped if already done)
      - Write meta.json with paths + primary caption
    Returns a list of metadata dicts and writes scenes_index.json.
    """
    # Import here to avoid circular deps when called from ingest_pipeline
    from caption_generator import get_best_caption

    pairs = find_kitti_pairs(max_scenes)
    print(f"Found {len(pairs)} valid LiDAR+image pairs (max_scenes={max_scenes})")

    metadata = []
    for bin_path, img_path, label_path, scene_id in pairs:
        scene_dir = SCENES_DIR / scene_id
        scene_dir.mkdir(exist_ok=True)

        # ── Point cloud ──
        pcd_out = scene_dir / "pointcloud.pcd"
        if not pcd_out.exists():
            bin_to_pcd(bin_path, pcd_out)

        # ── RGB image ──
        img_out = scene_dir / "image.jpg"
        if not img_out.exists():
            img = Image.open(img_path).convert("RGB")
            img.save(img_out, "JPEG", quality=95)

        # ── Rich caption from labels ──
        caption = "a driving scene"
        if label_path.exists():
            caption = get_best_caption(label_path)

        meta = {
            "scene_id":   scene_id,
            "pcd_path":   str(pcd_out),
            "image_path": str(img_out),
            "label_path": str(label_path) if label_path.exists() else None,
            "caption":    caption,
            "source":     "kitti",
        }
        with open(scene_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        metadata.append(meta)

    index_path = Path("data/kitti") / "scenes_index.json"
    with open(index_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built {len(metadata)} scenes → {SCENES_DIR}")
    print(f"Index saved → {index_path}")
    return metadata


def load_scene(scene_id: str) -> dict:
    """Load one scene's data from disk."""
    scene_dir = SCENES_DIR / scene_id
    with open(scene_dir / "meta.json") as f:
        meta = json.load(f)
    pcd   = o3d.io.read_point_cloud(meta["pcd_path"])
    image = Image.open(meta["image_path"]).convert("RGB")
    return {"scene_id": scene_id, "pcd": pcd, "image": image, "meta": meta}


if __name__ == "__main__":
    metadata = build_scene_dataset(max_scenes=7481)
    print("\nSample metadata entries:")
    for m in metadata[:3]:
        scene = load_scene(m["scene_id"])
        pts   = len(scene["pcd"].points)
        sz    = scene["image"].size
        print(f"  ✓ {m['scene_id']}: {pts} pts | {sz} | caption: {m['caption'][:60]}...")
