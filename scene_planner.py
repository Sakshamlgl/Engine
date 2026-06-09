import json
import re

import config


BEAT_ORDER = ["hook", "build_up", "discovery", "climax", "resolution"]
EMOTIONS = {
    "suspense",
    "shock",
    "sadness",
    "emotional",
    "dramatic",
    "uplifting",
    "funny",
    "mysterious",
}


def _has_groq_key():
    return (
        bool(config.GROQ_API_KEY)
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _clean_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in director response")
    return raw[start:end + 1]


def _split_sentences(script):
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    return [s.strip() for s in sentences if s.strip()]


def _target_scene_count(script):
    words = len(script.split())
    estimated = max(config.SCENE_PLAN_MIN, round(words / 24))
    return min(config.SCENE_PLAN_MAX, estimated)


def _chunk_sentences(sentences, count):
    if not sentences:
        return []

    groups = [[] for _ in range(min(count, len(sentences)))]
    for index, sentence in enumerate(sentences):
        groups[index % len(groups)].append(sentence)

    # Preserve original order after round-robin balancing.
    ordered = []
    cursor = 0
    remaining = sentences[:]
    for group in groups:
        size = len(group)
        ordered.append(remaining[cursor:cursor + size])
        cursor += size
    return [" ".join(group).strip() for group in ordered if group]


def _normalize_durations(scenes, target_duration):
    total = sum(max(1.0, float(scene.get("duration", 1))) for scene in scenes)
    if total <= 0:
        total = len(scenes) or 1

    remaining = float(target_duration)
    normalized = []
    for index, scene in enumerate(scenes):
        if index == len(scenes) - 1:
            duration = max(2.0, remaining)
        else:
            duration = max(2.0, round(float(scene.get("duration", 1)) / total * target_duration, 1))
            remaining -= duration
        normalized.append({**scene, "duration": round(duration, 1)})
    return normalized


def _infer_beat(index, total):
    if total <= 1:
        return "hook"
    if index == 0:
        return "hook"
    if index == total - 1:
        return "resolution"
    ratio = index / max(1, total - 1)
    if ratio < 0.38:
        return "build_up"
    if ratio < 0.68:
        return "discovery"
    return "climax"


def _infer_emotion(text, beat, niche):
    lower = text.lower()
    if any(word in lower for word in ["shocked", "screamed", "caught", "exposed", "betrayal"]):
        return "shock"
    if any(word in lower for word in ["cry", "tears", "heart", "alone", "lost"]):
        return "sadness"
    if any(word in lower for word in ["secret", "hidden", "mystery", "vanished", "unknown"]):
        return "suspense"
    if any(word in lower for word in ["funny", "laugh", "ridiculous", "awkward"]):
        return "funny"
    if niche.upper() == "D":
        return "mysterious"
    if niche.upper() == "B":
        return "uplifting" if beat == "resolution" else "dramatic"
    if niche.upper() == "C":
        return "dramatic"
    return "suspense" if beat in ("hook", "build_up") else "dramatic"


def _default_visual_goal(narration, niche, beat):
    lower = narration.lower()
    if niche.upper() == "B":
        if any(word in lower for word in ["space", "planet", "star", "galaxy"]):
            return "show cinematic space science imagery"
        if any(word in lower for word in ["cell", "bacteria", "brain", "body"]):
            return "show microscopic biology and research visuals"
        return "show a scientific discovery with documentary visuals"
    if niche.upper() == "C":
        return "show artificial intelligence technology and people reacting to it"
    if niche.upper() == "D":
        return "show historical mystery evidence and atmospheric ancient places"

    if any(word in lower for word in ["text", "phone", "message", "dm"]):
        return "show a person reading a shocking phone message"
    if any(word in lower for word in ["wedding", "bride", "husband", "wife"]):
        return "show a tense relationship or wedding betrayal moment"
    if any(word in lower for word in ["argument", "fight", "yelled", "confronted"]):
        return "show two people arguing in a tense indoor scene"
    if beat == "climax":
        return "show a shocked reaction during a dramatic reveal"
    return "show realistic social drama with tense body language"


def _queries_for_scene(narration, visual_goal, niche):
    """
    Fallback query generator — only used when Groq is unavailable.
    When Groq IS available, _ai_queries_for_scene() is called instead.
    """
    lower = (narration + " " + visual_goal).lower()
    if niche.upper() == "B":
        base = ["cinematic science research", "scientist laboratory", "documentary science visuals"]
    elif niche.upper() == "C":
        base = ["artificial intelligence technology", "person using computer", "futuristic data network"]
    elif niche.upper() == "D":
        base = ["ancient ruins documentary", "historical mystery evidence", "old archival footage"]
    elif "phone" in lower or "text" in lower or "message" in lower:
        base = ["person reading phone", "woman checking smartphone indoors", "shocked texting reaction"]
    elif "argument" in lower or "fight" in lower or "confront" in lower:
        base = ["couple arguing indoors", "tense conversation at home", "angry relationship argument"]
    elif "wedding" in lower or "bride" in lower:
        base = ["wedding drama reaction", "bride emotional indoors", "tense wedding moment"]
    else:
        base = ["dramatic person reaction", "tense indoor scene", "emotional face close up"]

    custom = re.sub(r"[^a-zA-Z0-9 ]+", " ", visual_goal).strip().lower()
    queries = [custom] if custom else []
    queries.extend(base)

    unique = []
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in unique:
            unique.append(query)
    return unique[:4]


def _ai_queries_for_scene(narration, visual_goal, emotion, beat, niche, full_script=""):
    """
    AI-powered query generation using Groq.

    Instead of keyword rules, this deeply understands the scene's role in the
    story and generates concrete, searchable stock queries.  Falls back to
    _queries_for_scene() on any error.
    """
    if not _has_groq_key():
        return _queries_for_scene(narration, visual_goal, niche)

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        niche_contexts = {
            "A": "Reddit drama / relationship story narration for YouTube Shorts",
            "B": "Science and nature documentary for YouTube Shorts",
            "C": "AI and technology news for YouTube Shorts",
            "D": "History mystery documentary for YouTube Shorts",
        }
        niche_context = niche_contexts.get(niche.upper(), niche_contexts["A"])

        script_excerpt = full_script[:400] if full_script else ""

        prompt = f"""You are a stock footage researcher for a {niche_context}.

Scene to find visuals for:
  Narration: "{narration}"
  Visual intent: "{visual_goal}"
  Emotion: {emotion}
  Story beat: {beat}
{"  Full script excerpt: " + script_excerpt if script_excerpt else ""}

Generate exactly 4 stock search queries for this scene.
Rules:
- Each query must describe something CONCRETELY PHOTOGRAPHABLE or FILMABLE that retains the core context of the story.
- Think like a stock footage librarian: what physical thing would a camera show?
- Be hyper-specific and context-aware: if the story is about a toxic online forum, search "computer screen showing angry comments" rather than "man at desk".
- Match the emotion: {emotion} scenes need {_emotion_visual_cues(emotion)}
- Try 4 different visual angles: reaction shot, wide scene, close detail, character moment
- Max 7 words per query
- No abstract concepts, no generic descriptions, only vivid visual descriptions tied to the context.

Return ONLY this JSON (no other text):
{{"queries": ["query1", "query2", "query3", "query4"]}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON found")
        parsed = json.loads(raw[start:end + 1])
        queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()]
        if len(queries) >= 2:
            return queries[:4]
        raise ValueError("Too few queries returned")

    except Exception as exc:
        # Silently fall back — this is called in a hot loop
        return _queries_for_scene(narration, visual_goal, niche)


def _emotion_visual_cues(emotion: str) -> str:
    """Helper: describe what makes a good visual for each emotion."""
    cues = {
        "shock":      "wide eyes, open mouth, sudden movement, high contrast lighting",
        "suspense":   "dark shadows, tense body language, looking over shoulder, waiting",
        "sadness":    "tears, downcast eyes, slouched posture, muted colors",
        "emotional":  "teary eyes, tight embrace, hand on face, warm light",
        "dramatic":   "intense eye contact, confrontation, strong shadows, close face",
        "uplifting":  "wide smile, arms raised, bright light, celebration",
        "mysterious": "fog, silhouette, obscured face, dark corridor",
        "funny":      "laughing, awkward pose, surprised face, casual setting",
    }
    return cues.get(emotion, "cinematic composition, expressive faces")


def _asset_types_for_scene(index, total, emotion):
    image_slots = round(total * config.PREFERRED_IMAGE_RATIO)
    if index < image_slots:
        return ["image", "video"]
    if emotion in ("shock", "dramatic"):
        return ["video", "image"]
    return ["image", "video"]


def _transition_for_emotion(emotion):
    if emotion == "shock":
        return "hard_cut"
    if emotion in ("sadness", "emotional"):
        return "fade"
    if emotion in ("suspense", "mysterious"):
        return "crossfade"
    return "hard_cut"


def _fallback_plan(script, post=None, niche="A", target_duration=None):
    target_duration = target_duration or config.TARGET_SECONDS
    sentences = _split_sentences(script)
    scene_count = _target_scene_count(script)
    chunks = _chunk_sentences(sentences, scene_count) or [script]

    scenes = []
    for index, narration in enumerate(chunks):
        beat = _infer_beat(index, len(chunks))
        emotion = _infer_emotion(narration, beat, niche)
        visual_goal = _default_visual_goal(narration, niche, beat)
        scenes.append({
            "scene_id": index + 1,
            "beat": beat,
            "narration": narration,
            "duration": max(3.0, len(narration.split()) / 3.1),
            "emotion": emotion,
            "visual_goal": visual_goal,
            "queries": _ai_queries_for_scene(
                narration, visual_goal, emotion, beat, niche, full_script=script
            ),
            "asset_types": _asset_types_for_scene(index, len(chunks), emotion),
            "character": "",
            "transition": _transition_for_emotion(emotion),
        })

    return _normalize_durations(scenes, target_duration)


def _validate_plan(raw_scenes, script, post=None, niche="A", target_duration=None):
    if not isinstance(raw_scenes, list):
        raise ValueError("Director response must be a JSON array")

    fallback = _fallback_plan(script, post=post, niche=niche, target_duration=target_duration)
    scenes = []

    for index, raw_scene in enumerate(raw_scenes[:config.SCENE_PLAN_MAX]):
        if not isinstance(raw_scene, dict):
            continue

        fallback_scene = fallback[min(index, len(fallback) - 1)]
        narration = str(raw_scene.get("narration") or fallback_scene["narration"]).strip()
        beat = str(raw_scene.get("beat") or fallback_scene["beat"]).strip().lower()
        emotion = str(raw_scene.get("emotion") or fallback_scene["emotion"]).strip().lower()
        visual_goal = str(raw_scene.get("visual_goal") or raw_scene.get("visual") or fallback_scene["visual_goal"]).strip()

        queries = raw_scene.get("queries")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            queries = fallback_scene["queries"]
        queries = [str(query).strip() for query in queries if str(query).strip()]
        if not queries:
            queries = _ai_queries_for_scene(narration, visual_goal, emotion, beat, niche)

        asset_types = raw_scene.get("asset_types") or raw_scene.get("media_types")
        if isinstance(asset_types, str):
            asset_types = [asset_types]
        if not isinstance(asset_types, list):
            asset_types = fallback_scene["asset_types"]
        asset_types = [
            media_type for media_type in asset_types
            if media_type in ("image", "video")
        ] or fallback_scene["asset_types"]

        if beat not in BEAT_ORDER:
            beat = fallback_scene["beat"]
        if emotion not in EMOTIONS:
            emotion = fallback_scene["emotion"]

        try:
            duration = float(raw_scene.get("duration", fallback_scene["duration"]))
        except (TypeError, ValueError):
            duration = fallback_scene["duration"]

        scenes.append({
            "scene_id": len(scenes) + 1,
            "beat": beat,
            "narration": narration,
            "duration": duration,
            "emotion": emotion,
            "visual_goal": visual_goal,
            "queries": queries[:4],
            "asset_types": asset_types[:2],
            "character": str(raw_scene.get("character") or fallback_scene.get("character", "")).strip(),
            "transition": str(raw_scene.get("transition") or _transition_for_emotion(emotion)).strip(),
        })

    if not scenes:
        return fallback
    return _normalize_durations(scenes, target_duration or config.TARGET_SECONDS)


def generate_scene_plan(script, post=None, niche="A", target_duration=None):
    """Return a director-style scene graph for the narrated short."""
    target_duration = target_duration or config.TARGET_SECONDS

    if not _has_groq_key():
        print("  No Groq key found for director planning; using local scene planner.")
        return _fallback_plan(script, post=post, niche=niche, target_duration=target_duration)

    try:
        from groq import Groq

        client = Groq(api_key=config.GROQ_API_KEY)
        topic = post.get("title", "") if post else ""
        content = post.get("content", "")[:900] if post else ""
        scene_count = _target_scene_count(script)

        prompt = f"""
You are the Director Agent for a YouTube Shorts generator.
Create a story-aware scene plan from this narration.

Topic: {topic}
Source context: {content}
Niche: {niche}
Target duration: {target_duration} seconds
Scene count: {scene_count}

Narration:
{script}

Return only valid JSON. The JSON must be an array of scene objects.
Each object must include:
- beat: one of hook, build_up, discovery, climax, resolution
- narration: the exact narration covered by this scene
- duration: estimated seconds
- emotion: one of suspense, shock, sadness, emotional, dramatic, uplifting, funny, mysterious
- visual_goal: semantic visual intention, not just a keyword
- queries: 3 to 4 stock search queries
- asset_types: preferred media types, e.g. ["image", "video"]
- character: exact, unique character name or object label (e.g. "John (protagonist)" or "Computer Screen"). NEVER use generic terms like "main subject".
- transition: hard_cut, crossfade, or fade

Use about 70 percent images and 30 percent videos.
Keep visuals realistic, specific, and directly searchable on stock sites.
Each query must be a concrete stock search term (e.g. "woman crying kitchen table", NOT "emotional moment").
Never use abstract or meta descriptions as queries.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1600,
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        scenes = json.loads(_clean_json(raw))
        return _validate_plan(scenes, script, post=post, niche=niche, target_duration=target_duration)
    except Exception as exc:
        print(f"  Director planning failed ({exc}); using local scene planner.")
        return _fallback_plan(script, post=post, niche=niche, target_duration=target_duration)