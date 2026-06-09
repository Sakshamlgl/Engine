"""
CLIP Ranker — Semantic Embedding Retrieval

Upgrade: Search → CLIP Embeddings → Semantic Similarity → Top Assets → Vision Verify

Instead of relying solely on keyword overlap (heuristic_score), CLIP encodes both
the scene's narrative intent and every candidate's text metadata into a shared
embedding space. Cosine similarity in that space is a far better proxy for visual
relevance than word matching.

How it works:
    Scene text  ──→ CLIP text encoder ──→ 512-d vector
    Candidate tags ──→ CLIP text encoder ──→ 512-d vector
    cosine(scene_vec, candidate_vec) → semantic_score 0..1

Why this matters more than prompt engineering:
    "woman devastated crying alone" and "female grief isolated" are near-identical
    in CLIP space but share zero keywords. The old ranker gave 0 overlap; CLIP
    gives 0.94 cosine similarity.

Integration:
    asset_ranker.rank_candidates() calls clip_rerank() after heuristic scoring
    when CLIP is available. The final score blends both signals.

Config (config.py / .env):
    CLIP_ENABLED = true          # master switch (default auto-detects sentence-transformers)
    CLIP_WEIGHT  = 0.55          # 0-1: how much CLIP influences final score vs heuristic

Dependencies (optional — degrades gracefully if missing):
    pip install sentence-transformers   # ~400 MB, CPU-friendly
    # or for true CLIP (requires torch):
    pip install open_clip_torch         # faster, more accurate

Memory: model is loaded once per process and cached (_MODEL singleton).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import config

# ── Config ────────────────────────────────────────────────────────────────────

CLIP_ENABLED = getattr(config, "CLIP_ENABLED", True)
CLIP_WEIGHT  = float(getattr(config, "CLIP_WEIGHT", 0.40))   # 0..1

# ── Model singleton (lazy load, thread-safe) ──────────────────────────────────

_MODEL     = None
_MODEL_LOCK = threading.Lock()
_MODEL_TYPE = None   # "openclip" | "sbert" | None


def _load_model():
    """
    Try to load the best available embedding model.
    Priority: open_clip_torch > sentence-transformers CLIP variant > None.
    """
    global _MODEL, _MODEL_TYPE

    with _MODEL_LOCK:
        if _MODEL is not None or _MODEL_TYPE is False:
            return _MODEL

        if getattr(config, "HF_TOKEN", None):
            try:
                import huggingface_hub
                huggingface_hub.login(token=config.HF_TOKEN)
                print("  🔑 Logged into HuggingFace Hub")
            except ImportError:
                pass

        # ── Option 1: open_clip (true CLIP, ~400 MB) ──────────────────────
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _MODEL = {"type": "openclip", "model": model, "tokenizer": tokenizer}
            _MODEL_TYPE = "openclip"
            print("  🔍 CLIP: loaded open_clip ViT-B-32 (high accuracy)")
            return _MODEL
        except Exception:
            pass

        # ── Option 2: sentence-transformers CLIP variant (~500 MB, CPU-friendly)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("clip-ViT-B-32")
            _MODEL = {"type": "sbert", "model": model}
            _MODEL_TYPE = "sbert"
            print("  🔍 CLIP: loaded sentence-transformers clip-ViT-B-32")
            return _MODEL
        except Exception:
            pass

        # ── Option 3: lightweight all-MiniLM as fallback (not true CLIP, still semantic)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _MODEL = {"type": "sbert_mini", "model": model}
            _MODEL_TYPE = "sbert_mini"
            print("  🔍 CLIP: loaded MiniLM semantic fallback (install open_clip for true CLIP)")
            return _MODEL
        except Exception:
            pass

        # No model available
        _MODEL_TYPE = False   # sentinel: don't retry
        return None


def _available() -> bool:
    """True if CLIP is enabled AND a model can be loaded."""
    if not CLIP_ENABLED:
        return False
    return _load_model() is not None


def _encode_texts(texts: list[str]) -> "list[list[float]] | None":
    """Encode a list of strings into embeddings. Returns None on failure."""
    model_bundle = _load_model()
    if model_bundle is None:
        return None

    try:
        mtype = model_bundle["type"]

        if mtype == "openclip":
            import torch
            import open_clip
            tokenizer = model_bundle["tokenizer"]
            model     = model_bundle["model"]
            # Truncate to 77 tokens (CLIP limit)
            texts_clipped = [t[:200] for t in texts]
            tokens = tokenizer(texts_clipped)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                # L2 normalise
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.numpy().tolist()

        elif mtype in ("sbert", "sbert_mini"):
            model = model_bundle["model"]
            vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.tolist()

    except Exception as exc:
        print(f"  ⚠️  CLIP encoding error: {exc}")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    # Already normalised — dot product is cosine
    return sum(x * y for x, y in zip(a, b))


# ── Public API ────────────────────────────────────────────────────────────────

def _scene_query_text(scene: dict) -> str:
    """Build a rich text description of what this scene needs visually."""
    parts = [
        scene.get("visual_goal", ""),
        scene.get("narration", ""),
        scene.get("emotion", ""),
        scene.get("beat", ""),
        " ".join(scene.get("queries", [])),
    ]
    return " ".join(p for p in parts if p).strip()[:512]


def _candidate_text(candidate: dict) -> str:
    """Build candidate text from all metadata fields."""
    parts = [
        candidate.get("query",       ""),
        candidate.get("tags",        ""),
        candidate.get("description", ""),
        candidate.get("page_url",    ""),
    ]
    return " ".join(p for p in parts if p).strip()[:300]


def clip_rerank(scene: dict, candidates: list[dict]) -> list[dict]:
    """
    Re-rank candidates using CLIP semantic similarity.

    Each candidate gets a new key ``clip_score`` (0-100).
    If CLIP is unavailable, candidates are returned unchanged (clip_score absent).

    Args:
        scene:      Scene dict with visual_goal, narration, emotion, queries, beat
        candidates: List of candidate dicts (from asset_fetcher / ranked by heuristic)

    Returns:
        Same list, each candidate optionally enriched with ``clip_score``.
    """
    if not candidates:
        return candidates

    if not _available():
        return candidates

    scene_text      = _scene_query_text(scene)
    candidate_texts = [_candidate_text(c) for c in candidates]

    # Encode all at once (batching is faster than one-by-one)
    all_texts = [scene_text] + candidate_texts
    vectors   = _encode_texts(all_texts)
    if vectors is None:
        return candidates

    scene_vec      = vectors[0]
    candidate_vecs = vectors[1:]

    for candidate, vec in zip(candidates, candidate_vecs):
        sim = _cosine(scene_vec, vec)          # -1..1
        # Map [-1, 1] → [0, 100]
        candidate["clip_score"] = round((sim + 1.0) / 2.0 * 100, 1)

    return candidates


def blend_score(heuristic: int, clip_score: float | None) -> int:
    """
    Blend heuristic score and CLIP score into a final ranking score.

    When CLIP is present:   final = (1-CLIP_WEIGHT)*heuristic + CLIP_WEIGHT*clip_score
    When CLIP is absent:    final = heuristic  (unchanged)
    """
    if clip_score is None:
        return heuristic
    raw = (1.0 - CLIP_WEIGHT) * heuristic + CLIP_WEIGHT * clip_score
    return max(0, min(100, int(round(raw))))


def has_clip() -> bool:
    """Returns True if CLIP is active for this run."""
    return _available()
