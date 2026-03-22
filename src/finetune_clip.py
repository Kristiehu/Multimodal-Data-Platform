"""
finetune_clip.py
----------------
Fine-tune CLIP on KITTI driving scenes with rich multi-variant captions.

Improvements over v1
--------------------
1. Caption richness   – uses caption_generator.py (spatial positions, depth
                        buckets, scene-type labels) instead of simple counts.
2. Data augmentation  – generates up to 3 distinct caption variants per scene
                        → triples the number of contrastive pairs.
3. Visual encoder     – unfreezes the last 2 transformer blocks so the image
                        encoder can also adapt (v1 froze it entirely).
4. Training data      – works with all 7481 scenes (or whatever is indexed).
5. Checkpoint         – saved to outputs/clip_finetuned.pt as before, so the
                        app and retrieval modules load it transparently.
"""

import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, "src")
from caption_generator import generate_captions

LABEL_DIR  = Path("data/kitti_raw/kitti-3d-object-detection-dataset/training/label_2")
SCENES_DIR = Path("data/kitti/scenes")
CKPT_PATH  = Path("outputs/clip_finetuned.pt")
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ── 1. Build (image_path, caption) pairs with augmentation ──────────────────

def build_training_pairs(max_variants: int = 3) -> list[dict]:
    """
    Build training pairs from the scenes index.

    For each scene that has a label file, generate up to `max_variants`
    distinct captions.  This multiplies the effective dataset size by up
    to 3×.
    """
    index_path = Path("data/kitti/scenes_index.json")
    if not index_path.exists():
        raise FileNotFoundError(
            "scenes_index.json not found. Run data_loader.py first:\n"
            "  PYTHONPATH=src python src/data_loader.py"
        )

    scenes = json.load(open(index_path))
    pairs: list[dict] = []

    for s in scenes:
        label_path = s.get("label_path")
        if not label_path or not Path(label_path).exists():
            # Fall back to the single caption stored in meta
            captions = [s.get("caption", "a driving scene")]
        else:
            captions = generate_captions(Path(label_path))[:max_variants]

        for cap in captions:
            pairs.append({
                "scene_id":   s["scene_id"],
                "image_path": s["image_path"],
                "caption":    cap,
            })

    print(f"Training pairs: {len(pairs)}  "
          f"(from {len(scenes)} scenes × up to {max_variants} captions)")
    if pairs:
        print("  Example pairs:")
        for p in pairs[:2]:
            print(f"    [{p['scene_id']}] {p['caption']}")
    return pairs


# ── 2. Dataset ───────────────────────────────────────────────────────────────

class KITTICaptionDataset(Dataset):
    def __init__(self, pairs: list[dict], preprocess):
        self.pairs      = pairs
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p     = self.pairs[idx]
        image = self.preprocess(Image.open(p["image_path"]).convert("RGB"))
        text  = clip.tokenize([p["caption"]], truncate=True)[0]
        return image, text


# ── 3. Contrastive loss ──────────────────────────────────────────────────────

def clip_contrastive_loss(
    image_feats: torch.Tensor,
    text_feats:  torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """Symmetric InfoNCE loss (same as CLIP's original training objective)."""
    logits  = image_feats @ text_feats.T / temperature
    labels  = torch.arange(len(logits), device=logits.device)
    loss_i  = F.cross_entropy(logits,   labels)
    loss_t  = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


# ── 4. Configure which parameters to train ──────────────────────────────────

def configure_trainable_params(model, n_visual_blocks: int = 2):
    """
    Freeze all parameters, then selectively unfreeze:
      • Entire text encoder (transformer + token embedding + positional embed)
      • Last `n_visual_blocks` transformer blocks of the visual encoder
      • Visual projection layer

    This is a pragmatic trade-off: the text encoder learns the driving
    vocabulary, while the last visual blocks learn driving-specific features
    without forgetting general visual representations.
    """
    # Start fully frozen
    for p in model.parameters():
        p.requires_grad = False

    # ── Text encoder ──
    for p in model.transformer.parameters():
        p.requires_grad = True
    model.token_embedding.weight.requires_grad = True
    model.positional_embedding.requires_grad   = True
    model.text_projection.requires_grad        = True
    if hasattr(model, "ln_final"):
        for p in model.ln_final.parameters():
            p.requires_grad = True

    # ── Last N visual transformer blocks ──
    visual = model.visual
    if hasattr(visual, "transformer"):
        blocks = visual.transformer.resblocks
        for block in list(blocks)[-n_visual_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    # Visual projection
    if hasattr(visual, "proj") and visual.proj is not None:
        visual.proj.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.1f}%)")


# ── 5. Training loop ─────────────────────────────────────────────────────────

def train(
    epochs:            int   = 15,
    batch_size:        int   = 32,
    lr:                float = 1e-6,
    n_visual_blocks:   int   = 2,
    max_variants:      int   = 3,
    accumulation_steps: int  = 4,
    resume_epoch:      int   = 8,   # set to last completed epoch to resume
):
    """
    Fine-tune CLIP on KITTI driving scenes.

    Parameters
    ----------
    epochs          : Total training epochs.
    batch_size      : Batch size.
    lr              : Peak learning rate.
    n_visual_blocks : Number of visual transformer blocks to unfreeze.
    max_variants    : Max caption variants per scene (data augmentation).
    resume_epoch    : Last completed epoch (0 = fresh start). The saved
                      checkpoint is loaded automatically and the LR scheduler
                      fast-forwards to the correct position.
    """
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model = model.float()    # MPS / CPU need float32; harmless on CUDA

    # ── Resume: load model + optimizer state ──
    OPT_CKPT = CKPT_PATH.parent / "optimizer_state.pt"
    if resume_epoch > 0 and CKPT_PATH.exists():
        state = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  Resumed model weights from checkpoint (epoch {resume_epoch} done)")

    pairs   = build_training_pairs(max_variants=max_variants)
    dataset = KITTICaptionDataset(pairs, preprocess)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    configure_trainable_params(model, n_visual_blocks=n_visual_blocks)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.98),
    )

    

    # Linear warm-up for first 10% of steps, cosine decay thereafter
    # total_steps must be in optimizer-step units (not batch units) when using
    # gradient accumulation, otherwise cosine decay barely moves.
    total_steps  = (epochs * len(loader)) // accumulation_steps
    warmup_steps = max(1, int(0.10 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    step      = resume_epoch * len(loader)   # fast-forward scheduler position

    # Resume optimizer state (preserves AdamW momentum & adaptive estimates)
    if resume_epoch > 0 and OPT_CKPT.exists():
        opt_state = torch.load(OPT_CKPT, map_location=DEVICE)
        optimizer.load_state_dict(opt_state["optimizer"])
        scheduler.load_state_dict(opt_state["scheduler"])
        print(f"  Resumed optimizer + scheduler state (no LR jump)")
    elif resume_epoch > 0:
        # No optimizer state saved — fast-forward scheduler as fallback
        for _ in range(step // accumulation_steps):
            scheduler.step()
        print(f"  Warning: no optimizer state found, fast-forwarded scheduler only")

    for epoch in range(resume_epoch, epochs):   # skip already-done epochs
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, texts) in enumerate(
                tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(DEVICE)
            texts  = texts.to(DEVICE)

            image_feats = model.encode_image(images)
            text_feats  = model.encode_text(texts)

            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            text_feats  = text_feats  / text_feats.norm(dim=-1, keepdim=True)

            # Divide loss so gradients accumulate to the correct scale
            loss = clip_contrastive_loss(image_feats, text_feats) / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps

            # Only update weights every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1

        avg_loss = total_loss / len(loader)
        cur_lr   = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  lr={cur_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CKPT_PATH)
            # Save optimizer + scheduler state alongside model weights
            torch.save({"optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch + 1},
                       OPT_CKPT)
            print(f"  ✓ Checkpoint saved → {CKPT_PATH}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return model


# ── 6. Rebuild embeddings after fine-tuning ──────────────────────────────────

def rebuild_embeddings_with_finetuned():
    """Re-embed all scenes and rebuild the FAISS index using the saved checkpoint."""
    from embed_images import embed_images
    from bev_projection import embed_bev, fuse_embeddings
    from vector_store import build_index

    print("\nRebuilding embeddings with fine-tuned model …")

    # Monkey-patch clip.load so downstream helpers automatically pick up weights
    _orig_load = clip.load

    def _patched_load(name, device=DEVICE):
        m, pre = _orig_load(name, device=device)
        state  = torch.load(CKPT_PATH, map_location=device)
        m.load_state_dict(state)
        m.eval()
        return m, pre

    clip.load = _patched_load
    try:
        embed_images(batch_size=32)
        embed_bev(batch_size=32)
        fuse_embeddings()
        build_index()
    finally:
        clip.load = _orig_load   # always restore

    print("Done — index rebuilt with fine-tuned embeddings.")


# ── 7. Quick post-training sanity check ──────────────────────────────────────

def quick_eval():
    """Run a few example queries and print top-3 results."""
    from retrieval import search

    _orig_load = clip.load

    def _patched_load(name, device=DEVICE):
        m, pre = _orig_load(name, device=device)
        state  = torch.load(CKPT_PATH, map_location=device)
        m.load_state_dict(state)
        m.eval()
        return m, pre

    clip.load = _patched_load
    try:
        for q in [
            "a scene with 3 cars",
            "pedestrians on the road",
            "empty road with no objects",
            "urban street with cars and cyclists",
        ]:
            search(q, k=3)
    finally:
        clip.load = _orig_load


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(
        epochs=15,
        batch_size=32,
        lr=1e-6,
        n_visual_blocks=2,
        max_variants=3,
        accumulation_steps=4,   # 实效 batch = 32 × 4 = 128
        resume_epoch=0,         # ← 断点续训：改成上次完成的 epoch 数，例如 8
    )
    rebuild_embeddings_with_finetuned()
    quick_eval()
