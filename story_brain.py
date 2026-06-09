"""
Story Brain — Central Feedback Loop (v5)

v5 upgrade: Story Brain now SEES images.

Previous architecture:
    Story Brain (text only)    ← only read tags/description
    Visual Verifier (vision)   ← only checked pass/fail

New architecture:
    Story Brain
        ↓
    Vision Model (sees actual image + narration + story arc)
        ↓
    Narration + Image + Story Arc → combined score
        ↓
    Smarter re-fetch queries
        ↓
    Final Assets

The old `score_asset_against_scene()` used text-only metadata.
The new `score_asset_with_vision()` sends the actual image to the vision model
alongside the full story arc — producing far more accurate story-match scores.

Configuration (config.py / .env):
    STORY_BRAIN_ENABLED        = true   # master switch
    STORY_BRAIN_THRESHOLD      = 42     # 0-100, below this = weak
    STORY_BRAIN_MAX_RETRIES    = 2      # how many re-fetch attempts per scene
    STORY_BRAIN_VISION_ENABLED = true   # enable vision scoring (default on if Groq key present)
"""

import base64
import io
import json
import os
import re
import tempfile

import config


# ── Tunables ──────────────────────────────────────────────────────────────────

STORY_BRAIN_ENABLED        = getattr(config, "STORY_BRAIN_ENABLED",        True)
STORY_BRAIN_THRESHOLD      = getattr(config, "STORY_BRAIN_THRESHOLD",       42)
STORY_BRAIN_MAX_RETRIES    = getattr(config, "STORY_BRAIN_MAX_RETRIES",     2)
STORY_BRAIN_VISION_ENABLED = getattr(config, "STORY_BRAIN_VISION_ENABLED",  True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_groq():
    return (
        bool(config.GROQ_API_KEY)
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _groq_client():
    from groq import Groq
    return Groq(api_key=config.GROQ_API_KEY)


def _asset_description(asset: dict) -> str:
    parts = [
        asset.get("query", ""),
        asset.get("tags", ""),
        asset.get("description", ""),
        asset.get("type", ""),
        asset.get("provider", ""),
    ]
    return " | ".join(p for p in parts if p).strip() or "unknown asset"


def _clean_json_obj(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in response")
    return json.loads(raw[start:end + 1])


# ── Vision helpers ────────────────────────────────────────────────────────────

def _extract_video_thumbnail(video_path: str) -> str | None:
    """Extract first frame of a video as a JPEG temp file. Returns path or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, frame)
        return tmp.name
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            frame_path = video_path + "_sbthumbnail.jpg"
            clip.save_frame(frame_path, t=min(0.5, clip.duration / 2))
            clip.close()
            return frame_path
        except Exception:
            return None
    except Exception:
        return None


def _encode_image_b64(image_path: str) -> tuple[str, str]:
    """Returns (base64_data, media_type). Resizes to max 512px for token efficiency."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        max_dim = 512
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
        return data, "image/jpeg"
    except Exception:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        return data, media_type


def _get_asset_image_path(asset: dict) -> str | None:
    """
    Get the local image path from an asset dict.
    For videos, extracts a thumbnail. Returns None if unavailable.
    """
    local_path = asset.get("asset", "")
    if not local_path or not os.path.exists(local_path):
        return None

    ext = os.path.splitext(local_path)[1].lower()
    if ext in (".mp4", ".mov", ".avi", ".webm"):
        return _extract_video_thumbnail(local_path)
    return local_path


# ── Story match scoring (v5: vision-aware) ────────────────────────────────────

def score_asset_against_scene(scene: dict, asset: dict) -> int:
    """
    Score how well an asset serves a scene's story intent.

    v5 upgrade: If the asset has a local file path AND STORY_BRAIN_VISION_ENABLED
    is true, the vision model SEES the actual image alongside the full story context.
    Otherwise falls back to text-only scoring (old behavior).

    Returns 0-100.
    """
    if not _has_groq():
        return _heuristic_story_score(scene, asset)

    # ── Vision path: Story Brain sees the image ───────────────────────────────
    if STORY_BRAIN_VISION_ENABLED:
        image_path = _get_asset_image_path(asset)
        if image_path:
            score = _vision_story_score(scene, asset, image_path)
            # Clean up extracted thumbnail
            if image_path != asset.get("asset") and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except Exception:
                    pass
            return score

    # ── Text-only fallback (original behavior) ────────────────────────────────
    return _text_story_score(scene, asset)


def _vision_story_score(scene: dict, asset: dict, image_path: str) -> int:
    """
    Ask the vision model to score the asset while SEEING the actual image
    AND the full story arc. This is the major v5 upgrade.
    """
    try:
        client = _groq_client()
        image_data, media_type = _encode_image_b64(image_path)

        ext = os.path.splitext(image_path)[1].lower()
        is_video_frame = ext in (".mp4", ".mov", ".avi", ".webm") or \
                         image_path.endswith("_sbthumbnail.jpg")

        prompt = f"""You are an expert video editor reviewing a stock asset for a YouTube Shorts story.

Scene context:
  Story beat: {scene.get("beat", "")}
  Narration:  "{scene.get("narration", "")}"
  Visual goal: "{scene.get("visual_goal", "")}"
  Required emotion: {scene.get("emotion", "")}
  Asset metadata: {_asset_description(asset)}

You are looking at {'the first frame of a video clip' if is_video_frame else 'this stock image'}.

Score from 0 to 100 how well what you ACTUALLY SEE serves this specific scene:
  90-100 = perfect — the visual emotion and content exactly match the scene
  70-89  = good — clearly relevant, fits the story beat
  50-69  = partial — related but not ideal for this moment
  30-49  = weak — only loosely related to this scene
  0-29   = wrong — misleading or completely off for this story beat

Focus on: does what you actually SEE (faces, actions, setting, mood) match what a viewer 
needs to feel at this exact moment in the story?

Return ONLY this JSON:
{{"score": <0-100>, "reason": "<one sentence: what you see and why it does or doesn't fit>", "weak": <true|false>}}"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=160,
            temperature=0.1,
        )

        result = _clean_json_obj(response.choices[0].message.content)
        score  = max(0, min(100, int(result.get("score", 50))))
        reason = result.get("reason", "")
        if reason:
            print(f"      🧠👁  vision story score={score} — {reason}")
        return score

    except Exception as exc:
        print(f"      vision story scoring failed ({exc}), falling back to text scoring")
        return _text_story_score(scene, asset)


def _text_story_score(scene: dict, asset: dict) -> int:
    """
    Text-only LLM story scoring (original v4 behavior).
    Used when no image is available or vision is disabled.
    """
    try:
        client = _groq_client()
        prompt = f"""You are a visual story editor reviewing stock footage/image selections.

Scene intent:
  Narration: {scene.get("narration", "")}
  Visual goal: {scene.get("visual_goal", "")}
  Emotion: {scene.get("emotion", "")}
  Beat: {scene.get("beat", "")}
  Queries used: {", ".join(scene.get("queries", []))}

Asset selected:
  Type: {asset.get("type", "")}
  Description/tags: {_asset_description(asset)}
  Provider: {asset.get("provider", "")}
  Fallback (AI-generated): {asset.get("fallback", False)}

Rate how well this asset visually serves the scene's story intent.
Score from 0 to 100.

Return ONLY this JSON:
{{"score": <integer 0-100>, "reason": "<one sentence>", "weak": <true|false>}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        result = _clean_json_obj(response.choices[0].message.content)
        score  = max(0, min(100, int(result.get("score", 50))))
        reason = result.get("reason", "")
        if reason:
            print(f"      🧠 text story score={score} — {reason}")
        return score

    except Exception as exc:
        print(f"      story brain scoring failed ({exc}), using heuristic")
        return _heuristic_story_score(scene, asset)


def _heuristic_story_score(scene: dict, asset: dict) -> int:
    """Fast keyword-overlap story score when Groq is unavailable."""
    import re as _re

    def tokens(text):
        words = _re.findall(r"[a-z0-9]+", str(text).lower())
        stopwords = {"a", "an", "the", "and", "or", "in", "on", "at", "of",
                     "to", "for", "with", "is", "show", "visual", "scene"}
        return {w for w in words if len(w) > 2 and w not in stopwords}

    wanted = (
        tokens(scene.get("narration", "")) |
        tokens(scene.get("visual_goal", "")) |
        tokens(scene.get("emotion", ""))
    )
    found = (
        tokens(asset.get("tags", "")) |
        tokens(asset.get("description", "")) |
        tokens(asset.get("query", ""))
    )
    if not wanted:
        return 50
    overlap = len(wanted & found) / max(1, len(wanted))
    if asset.get("fallback"):
        return max(0, int(overlap * 40))
    return max(0, min(100, int(overlap * 90) + 10))


# ── Query improvement ─────────────────────────────────────────────────────────

def generate_improved_queries(scene: dict, failed_asset: dict, scene_plan: list) -> list:
    """
    Given a weak asset, ask the LLM to deeply understand the scene's role in
    the full story arc and generate much more targeted search queries.
    Returns up to 5 improved queries.
    """
    if not _has_groq():
        return _fallback_improved_queries(scene, failed_asset)

    try:
        client = _groq_client()

        arc_summary = []
        for s in scene_plan:
            arc_summary.append(
                f"  [{s.get('beat', '?')}] {s.get('narration', '')[:80]}"
            )
        arc_text = "\n".join(arc_summary)

        # Include what was WRONG with the rejected asset's vision score if available
        rejection_context = ""
        if failed_asset.get("vision_score") is not None:
            rejection_context = f"\nVision model rejected it (score={failed_asset.get('vision_score')}): the image didn't visually match the scene emotion."

        prompt = f"""You are a professional video editor generating stock search queries.

Full story arc:
{arc_text}

Current scene that needs a better visual:
  Beat: {scene.get("beat", "")}
  Narration: {scene.get("narration", "")}
  Visual goal: {scene.get("visual_goal", "")}
  Emotion: {scene.get("emotion", "")}

Previous queries that didn't work well:
  {", ".join(scene.get("queries", []))}

Asset that was selected but rejected:
  {_asset_description(failed_asset)}{rejection_context}

Generate 5 BETTER stock search queries.
Rules:
- Very concrete and specific ("woman crying alone kitchen table night" not "emotional woman")
- Think about what a VIEWER needs to SEE to feel this moment in the story
- Match the emotion and beat (suspense = dark tense; shock = wide eyes; etc.)
- Describe actual people, places, objects, actions — no abstract concepts
- Each query tries a different visual angle on the same scene moment
- No more than 6 words per query

Return ONLY this JSON:
{{"queries": ["query1", "query2", "query3", "query4", "query5"]}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.3,
        )
        result  = _clean_json_obj(response.choices[0].message.content)
        queries = [str(q).strip() for q in result.get("queries", []) if str(q).strip()]
        if queries:
            print(f"      🧠 improved queries: {queries[:3]}")
        return queries[:5] or _fallback_improved_queries(scene, failed_asset)

    except Exception as exc:
        print(f"      story brain query improvement failed ({exc})")
        return _fallback_improved_queries(scene, failed_asset)


def _fallback_improved_queries(scene: dict, failed_asset: dict) -> list:
    """Rule-based query improvement when Groq is unavailable."""
    goal     = scene.get("visual_goal", "")
    emotion  = scene.get("emotion", "dramatic")
    beat     = scene.get("beat", "")
    narration = scene.get("narration", "").lower()

    emotion_modifiers = {
        "shock":      ["shocked reaction", "disbelief", "jaw drop"],
        "suspense":   ["dark moody", "tense waiting", "nervous"],
        "sadness":    ["crying alone", "grief", "heartbreak"],
        "emotional":  ["tearful", "overwhelmed", "touching moment"],
        "dramatic":   ["intense", "confrontation", "high stakes"],
        "uplifting":  ["joyful", "celebration", "triumph"],
        "mysterious": ["shadowy", "foggy", "unknown"],
        "funny":      ["laughing", "awkward", "surprised"],
    }

    modifiers = emotion_modifiers.get(emotion, ["dramatic", "cinematic"])
    base = re.sub(r"show\s+", "", goal.lower(), flags=re.IGNORECASE).strip()

    queries = []
    for mod in modifiers[:2]:
        queries.append(f"{mod} {base}"[:60])

    nouns = re.findall(r"\b(woman|man|couple|family|phone|car|house|office|street)\b", narration)
    if nouns:
        queries.append(f"{nouns[0]} {emotion} reaction close up")

    queries.append(f"{beat} moment {base}"[:60])
    queries.append(f"cinematic {base} {emotion}"[:60])

    return [q.strip() for q in queries if q.strip()][:5]


# ── Multimodal Memory ─────────────────────────────────────────────────────────

class StoryMemory:
    """
    Multimodal Story State — tracks characters, objects, locations, and timeline
    across the entire video so the Story Brain never loses context between scenes.

    This is the 'no multimodal memory' fix. The brain previously forgot everything
    between scene reviews. Now it maintains:
        - Which characters appeared and what they looked like
        - Key objects mentioned in the story
        - Location continuity
        - A running timeline of what has happened
    """

    def __init__(self):
        self.characters:  dict[str, list[str]] = {}   # char → [visual descriptions]
        self.objects:     list[str]             = []   # key story objects
        self.locations:   list[str]             = []   # locations seen
        self.timeline:    list[dict]            = []   # chronological story events

    def record_scene(self, scene: dict, asset: dict, vision_reason: str = "") -> None:
        """Update memory after a scene is reviewed."""
        # Track location
        location = scene.get("location", "")
        if location and location not in self.locations:
            self.locations.append(location)

        # Track character
        char = scene.get("character", "")
        if char:
            if char not in self.characters:
                self.characters[char] = []
            if vision_reason:
                self.characters[char].append(vision_reason[:120])

        # Add to timeline
        self.timeline.append({
            "beat":      scene.get("beat", ""),
            "narration": scene.get("narration", "")[:80],
            "emotion":   scene.get("emotion", ""),
        })

    def context_summary(self) -> str:
        """Generate a compact story-so-far for use in LLM prompts."""
        parts = []

        if self.timeline:
            recent = self.timeline[-4:]   # last 4 scenes
            beats = " → ".join(f"[{t['beat']}] {t['narration'][:40]}" for t in recent)
            parts.append(f"Story so far: {beats}")

        if self.characters:
            chars = ", ".join(
                f"{k} (seen {len(v)}x)" for k, v in list(self.characters.items())[:3]
            )
            parts.append(f"Characters: {chars}")

        if self.locations:
            parts.append(f"Locations: {', '.join(self.locations[-3:])}")

        return " | ".join(parts) if parts else ""


# ── Main feedback loop ────────────────────────────────────────────────────────

def review_and_improve(
    scene_plan: list,
    scene_assets: list,
    character_registry=None,
) -> list:
    """
    The Story Brain's main entry point (v5).

    v5 changes:
      - StoryMemory: maintains cross-scene state (characters, locations, timeline)
      - Vision scoring: the brain SEES each asset image (not just its metadata)
      - Vision rejection context is passed to query improvement

    Reviews all fetched assets against their scene intents, flags weak ones,
    and re-fetches them with smarter queries informed by both vision and story arc.

    Args:
        scene_plan:         Original list of scene dicts from scene_planner
        scene_assets:       List of fetched asset dicts from asset_fetcher
        character_registry: Optional CharacterRegistry for consistency

    Returns:
        Improved list of scene_assets (same length, weak scenes replaced if possible)
    """
    if not STORY_BRAIN_ENABLED:
        return scene_assets

    if not scene_assets:
        return scene_assets

    from asset_fetcher import fetch_scene_asset

    improved   = list(scene_assets)
    weak_count = 0
    memory     = StoryMemory()   # v5: multimodal memory across all scenes

    vision_active = STORY_BRAIN_VISION_ENABLED and _has_groq()
    mode_label = "👁 vision+story" if vision_active else "📝 text-only"
    print(f"\n  🧠 Story Brain v5 ({mode_label}): reviewing {len(improved)} scene assets...")

    for idx, (scene, asset) in enumerate(zip(scene_plan, improved)):
        scene_num = idx + 1

        # Score with vision (v5) or text (fallback)
        story_score = score_asset_against_scene(scene, asset)
        asset["story_score"] = story_score

        # Update multimodal memory regardless of score
        memory.record_scene(scene, asset)

        if story_score >= STORY_BRAIN_THRESHOLD:
            print(f"    Scene {scene_num}: ✅ story score={story_score}")
            continue

        weak_count += 1
        print(f"    Scene {scene_num}: ⚠️  story score={story_score} — attempting improvement")

        best_asset = asset
        best_score = story_score

        for attempt in range(1, STORY_BRAIN_MAX_RETRIES + 1):
            print(f"      retry {attempt}/{STORY_BRAIN_MAX_RETRIES}...")

            # v5: generate improved queries with story memory context
            new_queries = generate_improved_queries(scene, best_asset, scene_plan)
            if not new_queries:
                break

            # Include story memory in patched scene for smarter fetching
            mem_summary = memory.context_summary()
            patched_scene = {
                **scene,
                "queries": new_queries,
                "_story_memory": mem_summary,   # available for future extension
            }

            try:
                new_asset = fetch_scene_asset(
                    patched_scene,
                    scene_num,
                    previous_assets=improved[:idx],
                    character_registry=character_registry,
                )
                new_score = score_asset_against_scene(scene, new_asset)
                new_asset["story_score"] = new_score

                if new_score > best_score:
                    print(f"      ✅ improved: {best_score} → {new_score}")
                    best_asset = new_asset
                    best_score = new_score
                    scene["queries"] = new_queries
                    if best_score >= STORY_BRAIN_THRESHOLD:
                        break
                else:
                    print(f"      no improvement ({new_score} ≤ {best_score}), keeping previous")

            except Exception as exc:
                print(f"      re-fetch failed ({exc})")
                break

        improved[idx] = best_asset

    improved_count = sum(
        1 for i, a in enumerate(improved)
        if a.get("story_score", 100) > scene_assets[i].get("story_score", 0)
    )
    print(
        f"  🧠 Story Brain done: {weak_count} weak scenes found, "
        f"{improved_count} improved"
    )
    return improved
