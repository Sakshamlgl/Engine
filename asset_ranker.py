"""
Asset Ranker (v5) — Heuristic + CLIP Semantic + Optional LLM

v5 upgrade: CLIP semantic embeddings are now blended into the ranking score.

Scoring pipeline:
  1. heuristic_score()  — keyword overlap, resolution, duration, diversity (fast)
  2. clip_rerank()      — CLIP semantic similarity in embedding space (accurate)
  3. blend_score()      — weighted blend of heuristic + CLIP
  4. _llm_scores()      — optional LLM re-ranking (highest quality, costs tokens)

Why CLIP matters:
  "devastated woman alone kitchen" and "female grief isolated indoor" share
  zero keywords but near-identical CLIP embeddings. The heuristic gave 0;
  CLIP gives ~0.92 similarity. This alone closes the biggest relevance gap.
"""

import json
import re

import config


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with", "woman", "man", "person", "people", "show", "scene", "visual",
}


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", str(value).lower())
    return {word for word in words if len(word) > 2 and word not in STOPWORDS}


def _candidate_text(candidate):
    parts = [
        candidate.get("query", ""),
        candidate.get("tags", ""),
        candidate.get("description", ""),
        candidate.get("page_url", ""),
        candidate.get("provider", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _wanted_text(scene):
    values = [
        scene.get("visual_goal", ""),
        scene.get("narration", ""),
        " ".join(scene.get("queries", [])),
        " ".join(scene.get("asset_types", [])),
        scene.get("character", ""),
        scene.get("beat", ""),
        scene.get("emotion", ""),
    ]
    return " ".join(values)


def _resolution_score(candidate):
    width  = int(candidate.get("width")  or 0)
    height = int(candidate.get("height") or 0)
    if width <= 0 or height <= 0:
        return 4

    pixels = width * height
    score  = 0
    if pixels >= 1920 * 1080:
        score += 10
    elif pixels >= 1280 * 720:
        score += 7
    else:
        score += 3

    ratio = height / max(width, 1)
    if ratio >= 1.2:
        score += 4
    elif ratio >= 0.65:
        score += 2
    return score


def _duration_score(scene, candidate):
    if candidate.get("type") != "video":
        return 7

    duration = float(candidate.get("duration") or 0)
    wanted   = float(scene.get("duration") or 4)
    if duration <= 0:
        return 2
    if duration >= wanted:
        return 10
    if duration >= wanted * 0.65:
        return 7
    return 3


def _diversity_penalty(candidate, previous_assets):
    if not previous_assets:
        return 0

    current = _tokens(_candidate_text(candidate))
    if not current:
        return 0

    penalty = 0
    for previous in previous_assets[-3:]:
        previous_tokens = _tokens(_candidate_text(previous))
        if not previous_tokens:
            continue
        overlap = len(current & previous_tokens) / max(1, len(current | previous_tokens))
        if overlap > 0.45:
            penalty += 12
        elif overlap > 0.25:
            penalty += 6
    return penalty


def heuristic_score(scene, candidate, previous_assets=None, character_registry=None):
    wanted   = _tokens(_wanted_text(scene))
    found    = _tokens(_candidate_text(candidate))

    overlap   = len(wanted & found)
    relevance = 0 if not wanted else min(48, round((overlap / len(wanted)) * 80))

    exact_query_bonus = 0
    candidate_text    = _candidate_text(candidate).lower()
    for query in scene.get("queries", []):
        query_tokens = _tokens(query)
        if query_tokens and len(query_tokens & found) >= max(1, len(query_tokens) // 2):
            exact_query_bonus = max(exact_query_bonus, 14)
        if str(query).lower() in candidate_text:
            exact_query_bonus = max(exact_query_bonus, 18)

    type_bonus      = 0
    preferred_types = scene.get("asset_types", [])
    if candidate.get("type") in preferred_types[:1]:
        type_bonus = 14
    elif candidate.get("type") in preferred_types:
        type_bonus = 8

    char_bonus = 0
    if character_registry is not None:
        char_bonus = character_registry.consistency_bonus(
            scene.get("character", ""),
            candidate,
        )

    score = (
        relevance
        + exact_query_bonus
        + type_bonus
        + _resolution_score(candidate)
        + _duration_score(scene, candidate)
        - _diversity_penalty(candidate, previous_assets or [])
        + char_bonus
    )
    return max(0, min(100, int(score)))


def _has_llm_ranking():
    return (
        getattr(config, "LLM_ASSET_RANKING", False)
        and config.GROQ_API_KEY
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _llm_scores(scene, candidates):
    from groq import Groq

    client            = Groq(api_key=config.GROQ_API_KEY)
    compact_candidates = []
    for candidate in candidates[:8]:
        compact_candidates.append({
            "id":          candidate["rank_id"],
            "type":        candidate.get("type"),
            "query":       candidate.get("query"),
            "tags":        candidate.get("tags"),
            "description": candidate.get("description"),
            "duration":    candidate.get("duration"),
        })

    prompt = f"""
Score stock asset candidates for this scene from 0 to 100.
Scene visual goal: {scene.get("visual_goal")}
Scene emotion: {scene.get("emotion")}
Scene beat: {scene.get("beat")}
Scene narration: {scene.get("narration")}

Candidates:
{json.dumps(compact_candidates, indent=2)}

Return only JSON like:
{{"scores": [{{"id": "c1", "score": 92}}, ...]}}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.1,
    )
    raw   = response.choices[0].message.content.strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    parsed = json.loads(raw[start:end + 1])
    return {
        str(item["id"]): int(item["score"])
        for item in parsed.get("scores", [])
        if "id" in item and "score" in item
    }


def rank_candidates(scene, candidates, previous_assets=None, character_registry=None):
    """
    Full ranking pipeline (v5):
      1. Heuristic score every candidate
      2. CLIP semantic re-rank (blended in)
      3. Optional LLM re-rank (if LLM_ASSET_RANKING=true)
      4. Sort by final score descending
    """
    # ── Step 1: Heuristic ─────────────────────────────────────────────────────
    ranked = []
    for index, candidate in enumerate(candidates):
        prepared           = dict(candidate)
        prepared["rank_id"] = f"c{index + 1}"
        prepared["score"]   = heuristic_score(
            scene, prepared,
            previous_assets=previous_assets,
            character_registry=character_registry,
        )
        ranked.append(prepared)

    # ── Step 2: CLIP semantic re-ranking (v5 new) ─────────────────────────────
    try:
        from clip_ranker import clip_rerank, blend_score, has_clip
        if has_clip():
            ranked = clip_rerank(scene, ranked)
            for candidate in ranked:
                clip_s = candidate.get("clip_score")
                if clip_s is not None:
                    candidate["score"] = blend_score(candidate["score"], clip_s)
            print(f"    🔍 CLIP re-ranked {len(ranked)} candidates")
    except ImportError:
        pass   # clip_ranker not present — degrade gracefully
    except Exception as exc:
        print(f"    CLIP re-ranking skipped ({exc})")

    # ── Step 3: Optional LLM re-ranking ──────────────────────────────────────
    if _has_llm_ranking() and ranked:
        try:
            llm_scores = _llm_scores(scene, ranked)
            for candidate in ranked:
                llm_score = llm_scores.get(candidate["rank_id"])
                if llm_score is not None:
                    candidate["score"] = int(candidate["score"] * 0.35 + llm_score * 0.65)
        except Exception as exc:
            print(f"  Asset LLM ranking skipped ({exc}); using prior scores.")

    ranked.sort(key=lambda item: item.get("score", 0), reverse=True)
    return ranked
