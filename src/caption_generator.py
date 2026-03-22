"""
caption_generator.py
--------------------
Generates rich, spatially-aware text descriptions from KITTI label files.
Each scene gets up to 3 distinct caption variants for data augmentation.

KITTI label columns (space-separated):
  0  type          1  truncated   2  occluded   3  alpha
  4-7  2D bbox     8-10  3D dims (h,w,l)
  11-13  3D location (x_cam, y_cam, z_cam) — z_cam = forward depth (metres)
  14  rotation_y
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import NamedTuple

IGNORE_CLASSES = {"DontCare", "Misc"}

# Depth buckets (metres, camera z-axis = forward)
NEAR_MAX = 15.0
MID_MAX  = 30.0

# Lateral buckets (camera x-axis; negative = right side of scene in KITTI)
LATERAL_LEFT_THRESH  = -1.5   # x < -1.5  → right side (camera convention flipped)
LATERAL_RIGHT_THRESH =  1.5   # x >  1.5  → left side


class DetectedObject(NamedTuple):
    cls:      str
    depth:    float   # z in metres
    lateral:  float   # x in metres


def _parse_label_file(label_path: Path) -> list[DetectedObject]:
    """Parse one KITTI label_2 .txt file into a list of DetectedObjects."""
    objs: list[DetectedObject] = []
    try:
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in IGNORE_CLASSES:
                continue
            depth   = float(parts[13])   # z_cam
            lateral = float(parts[11])   # x_cam
            objs.append(DetectedObject(cls=cls, depth=depth, lateral=lateral))
    except Exception:
        pass
    return objs


def _depth_bucket(depth: float) -> str:
    if depth < NEAR_MAX:
        return "nearby"
    if depth < MID_MAX:
        return "at medium distance"
    return "far away"


def _lateral_bucket(x: float) -> str:
    """Map camera x → left/right/center (KITTI: negative x = right in world)."""
    if x > LATERAL_RIGHT_THRESH:
        return "on the left"
    if x < LATERAL_LEFT_THRESH:
        return "on the right"
    return "in the center"


def _scene_type(objs: list[DetectedObject]) -> str:
    """Heuristically classify the scene type from object mix."""
    if not objs:
        return "empty road"
    cls_counts = {}
    for o in objs:
        cls_counts[o.cls] = cls_counts.get(o.cls, 0) + 1
    total = len(objs)
    n_cars = cls_counts.get("Car", 0) + cls_counts.get("Van", 0)
    n_peds = cls_counts.get("Pedestrian", 0) + cls_counts.get("Person_sitting", 0)
    n_cyclists = cls_counts.get("Cyclist", 0)
    n_trucks = cls_counts.get("Truck", 0)
    n_trams = cls_counts.get("Tram", 0)

    if total == 0:
        return "empty road with no traffic"
    if n_trucks + n_trams > 2:
        return "urban road with heavy vehicles"
    if n_peds > 3 and n_cars < 2:
        return "pedestrian-heavy street"
    if n_cyclists > 2:
        return "road with cyclists"
    if n_cars > 6:
        return "busy city street"
    if n_cars > 3 and n_peds > 1:
        return "urban intersection"
    if n_cars > 2 and n_peds == 0:
        return "road with multiple vehicles"
    if n_cars == 1:
        return "quiet road"
    return "urban street scene"


def _object_summary(objs: list[DetectedObject]) -> str:
    """'3 cars, 1 pedestrian, 2 cyclists' style summary."""
    counts: dict[str, int] = {}
    for o in objs:
        counts[o.cls] = counts.get(o.cls, 0) + 1
    if not counts:
        return "no objects"
    parts = []
    priority = ["Car", "Van", "Pedestrian", "Cyclist", "Truck", "Tram", "Person_sitting"]
    ordered = sorted(counts.items(),
                     key=lambda kv: (priority.index(kv[0]) if kv[0] in priority else 99, -kv[1]))
    for cls, n in ordered:
        label = cls.lower()
        if cls == "Person_sitting":
            label = "seated person"
        if n > 1:
            label += "s" if not label.endswith("s") else ""
        parts.append(f"{n} {label}")
    return ", ".join(parts)


def _nearest_objects(objs: list[DetectedObject], n: int = 2) -> list[DetectedObject]:
    return sorted(objs, key=lambda o: o.depth)[:n]


# ── Caption variant generators ──────────────────────────────────────────────

def _caption_v1(objs: list[DetectedObject], scene_type: str) -> str:
    """
    Variant 1 – overview style.
    E.g.: "a busy city street with 4 cars and 2 pedestrians"
    """
    if not objs:
        return "an empty road with no vehicles or pedestrians"
    summary = _object_summary(objs)
    return f"a {scene_type} with {summary}"


def _caption_v2(objs: list[DetectedObject], scene_type: str) -> str:
    """
    Variant 2 – nearest-object focus.
    E.g.: "a car nearby on the right and a pedestrian in the center at medium distance"
    """
    if not objs:
        return "an open road with clear visibility and no obstacles"
    nearest = _nearest_objects(objs, n=3)
    parts = []
    for o in nearest:
        depth_str   = _depth_bucket(o.depth)
        lateral_str = _lateral_bucket(o.lateral)
        cls_str = o.cls.lower() if o.cls != "Person_sitting" else "seated person"
        parts.append(f"a {cls_str} {depth_str} {lateral_str}")
    core = " and ".join(parts)
    return f"{core}, driving through a {scene_type}"


def _caption_v3(objs: list[DetectedObject], scene_type: str) -> str:
    """
    Variant 3 – density / count-centric + distance context.
    E.g.: "3 nearby vehicles and 1 pedestrian far ahead on a city street"
    """
    if not objs:
        return "driving on an empty road with no traffic"

    near = [o for o in objs if o.depth < NEAR_MAX]
    mid  = [o for o in objs if NEAR_MAX <= o.depth < MID_MAX]
    far  = [o for o in objs if o.depth >= MID_MAX]

    segments = []
    if near:
        segments.append(f"{len(near)} nearby object{'s' if len(near) > 1 else ''}")
    if mid:
        segments.append(f"{len(mid)} object{'s' if len(mid) > 1 else ''} at medium range")
    if far:
        segments.append(f"{len(far)} distant object{'s' if len(far) > 1 else ''}")
    distribution = ", ".join(segments) if segments else "several objects"
    return f"a {scene_type} with {distribution} visible"


def generate_captions(label_path: Path) -> list[str]:
    """
    Generate up to 3 distinct caption variants for a KITTI scene.
    Always returns a list with at least 1 caption.
    """
    objs = _parse_label_file(label_path)
    scene_type = _scene_type(objs)

    v1 = _caption_v1(objs, scene_type)
    v2 = _caption_v2(objs, scene_type)
    v3 = _caption_v3(objs, scene_type)

    # Deduplicate while preserving order
    seen: set[str] = set()
    captions: list[str] = []
    for cap in [v1, v2, v3]:
        if cap not in seen:
            seen.add(cap)
            captions.append(cap)
    return captions


def get_best_caption(label_path: Path) -> str:
    """Return the single richest (variant 1) caption for a scene."""
    return generate_captions(label_path)[0]


# ── CLI demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    label_dir = Path("data/kitti_raw/kitti-3d-object-detection-dataset/training/label_2")
    sample_files = sorted(label_dir.glob("*.txt"))[:10]
    for lf in sample_files:
        print(f"\n=== {lf.stem} ===")
        for i, cap in enumerate(generate_captions(lf), 1):
            print(f"  V{i}: {cap}")
