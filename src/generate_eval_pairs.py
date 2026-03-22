"""
generate_eval_pairs.py
----------------------
Build a proper semantic evaluation set for Recall@k measurement.

Problem with the previous eval
-------------------------------
The old evaluation used "adjacent frame" as ground truth
(e.g. query "road with cars" → expected scene_000001).
Adjacent frames in KITTI often look very similar regardless of query content,
making the eval misleading.  A single hit from 5 hard-coded pairs also has
very high variance.

This script creates semantically-grounded (query, scene_id) pairs by:
1. Parsing all KITTI label files to extract scene-level statistics.
2. For each semantic category (e.g. "scene with many pedestrians"), finding
   the top-N matching scenes according to the label statistics.
3. Generating natural-language query strings for each category.
4. Saving the pairs to outputs/eval_pairs.json.

Recall@5 is then measured as: for each query, is the matched scene in the
top-5 FAISS results?

Usage
-----
    PYTHONPATH=src python src/generate_eval_pairs.py
"""

import json
import random
from pathlib import Path
from typing import Any

LABEL_DIR  = Path("data/kitti_raw/kitti-3d-object-detection-dataset/training/label_2")
INDEX_PATH = Path("data/kitti/scenes_index.json")
OUT_PATH   = Path("outputs/eval_pairs.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

IGNORE = {"DontCare", "Misc"}

random.seed(42)


# ── 1. Parse label statistics ────────────────────────────────────────────────

def parse_scene_stats(label_path: Path) -> dict[str, Any]:
    """Return object counts and depth/position summaries for one scene."""
    counts: dict[str, int] = {}
    depths: list[float]    = []

    try:
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in IGNORE:
                continue
            counts[cls] = counts.get(cls, 0) + 1
            depths.append(float(parts[13]))   # z_cam
    except Exception:
        pass

    n_cars  = counts.get("Car", 0)  + counts.get("Van", 0)
    n_peds  = counts.get("Pedestrian", 0) + counts.get("Person_sitting", 0)
    n_cyc   = counts.get("Cyclist", 0)
    n_truck = counts.get("Truck", 0)
    total   = sum(counts.values())
    avg_d   = sum(depths) / len(depths) if depths else 0.0
    min_d   = min(depths) if depths else 0.0

    return {
        "n_cars": n_cars, "n_peds": n_peds, "n_cyclists": n_cyc,
        "n_trucks": n_truck, "total": total,
        "avg_depth": avg_d, "min_depth": min_d,
        "counts": counts,
    }


def build_stats_index(scenes: list[dict]) -> list[dict]:
    """Attach parsed statistics to each scene entry."""
    enriched = []
    for s in scenes:
        lp = s.get("label_path")
        if lp and Path(lp).exists():
            stats = parse_scene_stats(Path(lp))
        else:
            stats = {"n_cars": 0, "n_peds": 0, "n_cyclists": 0,
                     "n_trucks": 0, "total": 0, "avg_depth": 0.0,
                     "min_depth": 0.0, "counts": {}}
        enriched.append({**s, **stats})
    return enriched


# ── 2. Category-based pair generators ────────────────────────────────────────

def _top_n(enriched: list[dict], key_fn, n: int = 5) -> list[dict]:
    return sorted(enriched, key=key_fn, reverse=True)[:n]


def _sample(pool: list[dict], n: int) -> list[dict]:
    return random.sample(pool, min(n, len(pool)))


def generate_pairs(enriched: list[dict]) -> list[dict]:
    """Return a list of {query, scene_id, category} dicts."""
    pairs: list[dict] = []

    def add(query: str, scene: dict, category: str):
        pairs.append({"query": query, "scene_id": scene["scene_id"],
                      "category": category})

    # ── many cars ──
    for s in _top_n(enriched, lambda x: x["n_cars"], n=10):
        add("a busy road with many cars", s, "many_cars")
        add("heavy vehicle traffic on a street", s, "many_cars")

    # ── pedestrians ──
    ped_scenes = [s for s in enriched if s["n_peds"] >= 2]
    for s in _sample(ped_scenes, 10):
        add("pedestrians walking on the road", s, "pedestrians")
        add("a street with people crossing", s, "pedestrians")

    # ── cyclists ──
    cyc_scenes = [s for s in enriched if s["n_cyclists"] >= 1]
    for s in _sample(cyc_scenes, 8):
        add("a road with a cyclist nearby", s, "cyclists")
        add("bicycle traffic on the street", s, "cyclists")

    # ── trucks / heavy vehicles ──
    truck_scenes = [s for s in enriched if s["n_trucks"] >= 1]
    for s in _sample(truck_scenes, 8):
        add("a truck driving on the road", s, "trucks")
        add("heavy vehicle or bus ahead", s, "trucks")

    # ── empty / sparse ──
    empty_scenes = [s for s in enriched if s["total"] == 0]
    for s in _sample(empty_scenes, 8):
        add("an empty road with no vehicles", s, "empty")
        add("driving on a clear road with no obstacles", s, "empty")

    # ── mixed urban (cars + peds) ──
    urban = [s for s in enriched if s["n_cars"] >= 2 and s["n_peds"] >= 1]
    for s in _sample(urban, 10):
        add("urban street with cars and pedestrians", s, "urban_mixed")
        add("busy city intersection with multiple objects", s, "urban_mixed")

    # ── nearby objects (min_depth < 10m) ──
    near = [s for s in enriched if s["min_depth"] > 0 and s["min_depth"] < 10.0]
    for s in _sample(near, 8):
        add("a car very close in front", s, "close_objects")
        add("an object right ahead on the road", s, "close_objects")

    # ── distant-only scenes (avg_depth > 25m) ──
    distant = [s for s in enriched if s["avg_depth"] > 25.0 and s["total"] > 0]
    for s in _sample(distant, 8):
        add("vehicles far ahead on an open road", s, "distant_objects")
        add("open highway with objects in the distance", s, "distant_objects")

    print(f"Generated {len(pairs)} eval pairs from {len(set(p['scene_id'] for p in pairs))} "
          f"distinct scenes across {len(set(p['category'] for p in pairs))} categories.")
    return pairs


# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} not found. Run the ingest pipeline first:\n"
            "  PYTHONPATH=src python src/ingest_pipeline.py"
        )

    scenes   = json.load(open(INDEX_PATH))
    enriched = build_stats_index(scenes)

    pairs = generate_pairs(enriched)

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} eval pairs → {OUT_PATH}")

    # Summary by category
    by_cat: dict[str, int] = {}
    for p in pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    print("\nPairs per category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<20} {n}")


if __name__ == "__main__":
    main()
