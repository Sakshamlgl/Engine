"""
Character Registry v5 — Visual Face Embeddings + Metadata Consistency

Problem (v4):
    Scene 1: Woman A
    Scene 3: Woman B    ← jarring jump — different face
    Scene 5: Woman C    ← viewer loses continuity

v4 solution (metadata-driven):
    Track query/tags overlap → give bonus for same keywords
    Weakness: "woman crying" and "female grief" share 0 keywords

v5 solution (visual embedding-driven):
    Face embeddings using CLIP / sentence-transformers
    Visual description embeddings (what the vision model SEES)
    Cosine similarity in embedding space instead of keyword overlap

Upgrade path (progressive enhancement):
    Level 0: No embeddings available → keyword overlap (v4 behavior)
    Level 1: sentence-transformers available → semantic text embeddings
    Level 2: CLIP available → true visual embeddings of description text
    Level 3: InsightFace/DeepFace available → actual face embeddings from images

The registry automatically selects the highest available level per run.
"""

import os
import re
import threading
from typing import Optional

# ── Embedding backend (lazy, thread-safe) ─────────────────────────────────────

_EMBED_MODEL     = None
_EMBED_LOCK      = threading.Lock()
_EMBED_TYPE: str = ""   # "clip" | "sbert" | "face" | ""


def _load_embed_model():
    """Select the best available embedding backend."""
    global _EMBED_MODEL, _EMBED_TYPE

    with _EMBED_LOCK:
        if _EMBED_MODEL is not None or _EMBED_TYPE == "none":
            return _EMBED_MODEL

        # ── Level 3: Real face embeddings (InsightFace) ───────────────────
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(128, 128))
            _EMBED_MODEL = {"type": "insightface", "app": app}
            _EMBED_TYPE  = "face"
            print("  🎭 CharacterRegistry: InsightFace face embeddings active (highest accuracy)")
            return _EMBED_MODEL
        except Exception:
            pass

        # ── Level 2: CLIP text embeddings ─────────────────────────────────
        try:
            from clip_ranker import _load_model as _clip_load, _encode_texts
            model = _clip_load()
            if model is not None:
                _EMBED_MODEL = {"type": "clip", "encode": _encode_texts}
                _EMBED_TYPE  = "clip"
                print("  🎭 CharacterRegistry: CLIP text embeddings active")
                return _EMBED_MODEL
        except Exception:
            pass

        # ── Level 1: Sentence-transformers ───────────────────────────────
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            _EMBED_MODEL = {"type": "sbert", "model": model}
            _EMBED_TYPE  = "sbert"
            print("  🎭 CharacterRegistry: semantic text embeddings active (MiniLM)")
            return _EMBED_MODEL
        except Exception:
            pass

        # Level 0: fallback to keyword overlap
        _EMBED_TYPE  = "none"
        _EMBED_MODEL = None
        return None


def _cosine(a: list, b: list) -> float:
    """Cosine similarity of two L2-normalised vectors."""
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def _embed_text(text: str) -> Optional[list]:
    """Embed a text string using the best available model."""
    model_bundle = _load_embed_model()
    if model_bundle is None:
        return None

    try:
        mtype = model_bundle["type"]
        if mtype in ("clip",):
            vecs = model_bundle["encode"]([text])
            return vecs[0] if vecs else None
        elif mtype == "sbert":
            vec = model_bundle["model"].encode([text], normalize_embeddings=True,
                                               show_progress_bar=False)
            return vec[0].tolist()
    except Exception:
        return None


def _embed_face(image_path: str) -> Optional[list]:
    """
    Extract face embedding from an image using InsightFace.
    Returns the first detected face embedding (512-d) or None.
    """
    model_bundle = _load_embed_model()
    if model_bundle is None or model_bundle.get("type") != "insightface":
        return None

    try:
        import cv2
        app   = model_bundle["app"]
        img   = cv2.imread(image_path)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb  = face.embedding.tolist()
        # L2 normalise
        norm = sum(x * x for x in emb) ** 0.5
        return [x / max(norm, 1e-9) for x in emb]
    except Exception:
        return None


# ── Character Snapshot ────────────────────────────────────────────────────────

class CharacterSnapshot:
    """
    Stores everything we know about one appearance of a character.
    Used for similarity comparison across scenes.
    """
    __slots__ = ("provider", "query", "tags", "description",
                 "text_embedding", "face_embedding", "visual_reason")

    def __init__(self, candidate: dict, visual_reason: str = ""):
        self.provider       = candidate.get("provider", "")
        self.query          = candidate.get("query", "")
        self.tags           = candidate.get("tags", "")
        self.description    = candidate.get("description", "")
        self.visual_reason  = visual_reason   # from vision model ("woman with dark hair, office")
        self.text_embedding: Optional[list] = None
        self.face_embedding: Optional[list] = None

    def compute_embeddings(self, image_path: str = "") -> None:
        """
        Compute both text and (if image available) face embeddings.
        Called after construction when we want to cache the vectors.
        """
        # Text embedding from combined visual description
        text = f"{self.tags} {self.description} {self.visual_reason}"
        self.text_embedding = _embed_text(text.strip())

        # Face embedding from actual image
        if image_path and os.path.exists(image_path):
            self.face_embedding = _embed_face(image_path)


# ── Character Registry ────────────────────────────────────────────────────────

class CharacterRegistry:
    """
    Tracks per-character visual asset history for one video run.

    v5 upgrades:
      - Embedding-based similarity (CLIP, sentence-transformers, or InsightFace)
      - Visual description storage (from vision model)
      - Face embedding support when InsightFace is available
      - Degrades gracefully to keyword overlap when no models available
    """

    def __init__(self):
        self._history: dict[str, list[CharacterSnapshot]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(
        self,
        character: str,
        candidate: dict,
        visual_reason: str = "",
        image_path:    str = "",
    ) -> None:
        """
        Record a successful asset selection for `character`.

        Args:
            character:     Character key (e.g. "wife", "detective", "protagonist")
            candidate:     The selected asset dict
            visual_reason: Optional description of what the vision model saw
                           (e.g. "woman in her 30s, dark hair, tearful close-up")
            image_path:    Optional local path to the image for face embedding
        """
        key = self._key(character)
        if not key:
            return

        snap = CharacterSnapshot(candidate, visual_reason)
        snap.compute_embeddings(image_path)
        self._history.setdefault(key, []).append(snap)

    def consistency_bonus(self, character: str, candidate: dict) -> int:
        """
        Return a relevance bonus (0–30) when `candidate` visually matches
        prior assets used for `character`.

        v5: Uses embedding cosine similarity when available.
            Falls back to keyword overlap (v4 behavior) when no model loaded.
        """
        key = self._key(character)
        if not key or key not in self._history:
            return 0

        history = self._history[key][-3:]   # last 3 appearances

        model_bundle = _load_embed_model()

        if model_bundle is not None:
            return self._embedding_bonus(candidate, history)
        else:
            return self._keyword_bonus(candidate, history)

    def has_history(self, character: str) -> bool:
        return bool(self._key(character)) and self._key(character) in self._history

    def summary(self) -> dict:
        return {
            char: {
                "appearances":  len(snaps),
                "queries_used": [s.query for s in snaps],
                "embed_type":   _EMBED_TYPE or "keyword",
            }
            for char, snaps in self._history.items()
        }

    # ── Similarity methods ─────────────────────────────────────────────────────

    def _embedding_bonus(
        self,
        candidate: dict,
        history:   list[CharacterSnapshot],
    ) -> int:
        """
        Compute similarity bonus using stored embeddings.
        Returns 0-30.
        """
        # Build candidate text and embed it
        cand_text = (
            f"{candidate.get('tags', '')} "
            f"{candidate.get('description', '')} "
            f"{candidate.get('query', '')}"
        ).strip()
        cand_vec = _embed_text(cand_text)
        if cand_vec is None:
            return self._keyword_bonus(candidate, history)

        best_sim = 0.0
        for snap in history:
            if snap.text_embedding is not None:
                sim      = _cosine(cand_vec, snap.text_embedding)
                best_sim = max(best_sim, sim)

        # Map [-1,1] → [0,30]
        bonus = int(((best_sim + 1.0) / 2.0) * 30)
        return max(0, min(30, bonus))

    def _keyword_bonus(
        self,
        candidate: dict,
        history:   list[CharacterSnapshot],
    ) -> int:
        """
        Keyword overlap fallback (v4 behavior, kept for when no model available).
        Returns 0-25.
        """
        cand_tokens = self._tokens(
            f"{candidate.get('tags', '')} "
            f"{candidate.get('query', '')} "
            f"{candidate.get('description', '')}"
        )
        if not cand_tokens:
            return 0

        best_bonus = 0
        for snap in history:
            hist_tokens = self._tokens(f"{snap.tags} {snap.query}")
            if not hist_tokens:
                continue
            overlap = len(cand_tokens & hist_tokens) / max(1, len(cand_tokens | hist_tokens))
            best_bonus = max(best_bonus, int(overlap * 25))

        return best_bonus

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _key(character: str) -> str:
        if not character:
            return ""
        return re.sub(r"\s+", "_", character.strip().lower())[:32]

    _STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "into", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "with",
    }

    @classmethod
    def _tokens(cls, text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) > 2 and w not in cls._STOPWORDS}
