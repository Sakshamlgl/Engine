# v5 Upgrade — What Changed and Why

## 1. CLIP Retrieval (`clip_ranker.py`) ★ Highest ROI

**Problem:** Keyword ranker gave 0 score to semantically identical queries.  
`"devastated woman alone kitchen"` ≠ `"female grief isolated indoor"` (0 keyword overlap).

**Solution:** CLIP semantic embeddings. Both phrases map to nearly identical vectors in CLIP space (~0.93 cosine similarity), so the ranker correctly identifies them as the same visual concept.

**How it works:**
```
Search → Heuristic Score → CLIP Re-rank (blend) → LLM Re-rank (optional) → Top Asset
```

**Configuration:**
```
CLIP_ENABLED=true
CLIP_WEIGHT=0.55    # 0=heuristic only, 1=CLIP only
```

**Dependencies (auto-detected, best available used):**
```
pip install sentence-transformers   # Level 1: semantic text (easiest)
pip install open_clip_torch         # Level 2: true CLIP (more accurate)
```

---

## 2. Story Brain Vision (`story_brain.py`) ★ Major quality jump

**Problem:** Story Brain scored assets using only text metadata (tags, description).  
It never saw the actual image.

**Old flow:**
```
Story Brain (tags/description only) ──→ score
Visual Verifier (image) ──→ pass/fail
```

**New flow:**
```
Story Brain
    ↓
Vision Model (sees: actual image + narration + full story arc)
    ↓
Combined score (what it sees vs what story needs)
    ↓
Smarter re-fetch queries (informed by what was visually wrong)
```

**Key change:** `score_asset_against_scene()` now calls `_vision_story_score()` which sends the actual image to `llama-3.2-11b-vision` (same Groq key, no extra cost). Story Brain and Visual Verifier are now unified — the brain sees what the verifier sees.

---

## 3. Multimodal Story Memory (`story_brain.StoryMemory`)

**Problem:** Story Brain forgot everything between scene reviews.

**Solution:** `StoryMemory` class maintains across all scenes:
- **Characters:** who appeared and what the vision model described
- **Locations:** where the story has been set  
- **Timeline:** chronological beat-by-beat story events

This context is passed to query improvement — the LLM now knows "this is scene 5, after [hook] she found the note → [build_up] she called her sister → [discovery] the sister confessed" when generating replacement queries.

---

## 4. Editing Intelligence (`editing_intelligence.py`) ★ New module

**Problem:** All shot editing was rule-based. No system decided:
- Use reaction shot first (suspense)?
- Smash cut here (shock)?
- Hold 14 frames here (discovery)?
- Ken Burns zoom-out (resolution)?

**Solution:** `annotate_edit_decisions()` analyses every shot and assigns:

| Decision | Controls |
|---|---|
| `cut_style` | `hard` / `soft` / `smash` — how to enter this shot |
| `pacing` | `slow` / `normal` / `fast` — duration multiplier |
| `ken_direction` | `in` / `out` / `left` / `right` — zoom direction |
| `ken_intensity` | `1–3` — zoom aggressiveness |
| `hold_frames` | `0–20` — still frames at start (0=action, 14=reveal) |
| `reaction_first` | `bool` — swap shot order for suspense |
| `audio_duck` | `bool` — duck music for dialogue moments |

**Beat profiles:**
- `hook` → hard cut, fast, zoom in intensity 2
- `discovery` → soft cut, slow, zoom out (reveal), hold 12 frames, reaction first
- `climax` → smash cut, fast, zoom in intensity 3
- `resolution` → soft cut, slow, zoom out, audio duck

LLM is called for `climax` and `discovery` beats only (saves tokens, highest return).

**Integration:** `video_maker.py` reads `shot["edit"]` for all directives.  
Helper functions `get_ken_burns_params(shot)` and `get_transition(shot)` translate directives to existing video_maker parameters.

---

## 5. Character Embeddings (`character_registry.py`) ★ Embedding upgrade

**Problem:** Character consistency was metadata-driven (keyword overlap).  
`"woman crying"` and `"female grief isolated"` share 0 keywords → 0 consistency bonus.

**Solution:** Progressive enhancement — auto-selects the best available backend:

| Level | Backend | Accuracy |
|---|---|---|
| 3 | InsightFace | Real face embeddings from actual images |
| 2 | CLIP (open_clip / sentence-transformers) | Semantic visual description embeddings |
| 1 | MiniLM (sentence-transformers) | Semantic text embeddings |
| 0 | Keyword overlap | v4 behavior (no dependencies) |

The registry now stores `CharacterSnapshot` objects with both text and face embeddings, enabling cosine-similarity comparison instead of keyword matching.

**Optional upgrade for true face consistency:**
```
pip install insightface
pip install opencv-python-headless
```

---

## Files Changed

| File | Change |
|---|---|
| `clip_ranker.py` | **NEW** — CLIP semantic retrieval |
| `editing_intelligence.py` | **NEW** — Beat-aware editing decisions |
| `story_brain.py` | **UPGRADED** — Vision scoring + StoryMemory |
| `asset_ranker.py` | **UPGRADED** — CLIP integration in ranking pipeline |
| `character_registry.py` | **UPGRADED** — Embedding-based consistency |
| `main.py` | **UPDATED** — Phase 4d wired in, docstring updated |
| `config.py` | **UPDATED** — 5 new feature flags |
| `requirements.txt` | **UPDATED** — sentence-transformers added |
| `.env.example` | **UPDATED** — New flags documented |
