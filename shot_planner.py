"""
Shot Planner — Multi-Shot Scene Expansion

Transforms the scene plan from:

    1 scene = 1 asset

into professional short-form video shot structure:

    scene
      ├── wide shot       (establishes context)
      ├── close-up        (emotion / detail)
      ├── reaction shot   (character response)  [optional]
      └── detail shot     (key object/moment)   [optional]

This is how professional YouTube Shorts, TikToks, and documentaries are cut.
Each "shot" becomes an asset search target with its own tailored query.

The planner is beat-aware:
  hook       → 2 shots (wide + close, tight pacing)
  build_up   → 2 shots (context + tension)
  discovery  → 3 shots (wide + close + reaction)
  climax     → 3-4 shots (wide + close + reaction + detail)
  resolution → 2 shots (close + wide, emotional close-out)

Configuration (config.py / .env):
    SHOT_PLANNING_ENABLED = true
    SHOT_PLANNING_MAX     = 4    # max shots per scene (default 4)

Usage:
    from shot_planner import expand_scene_plan
    expanded = expand_scene_plan(scene_plan)
    # expanded has more scenes but same total duration
"""

import json
import re

import config

SHOT_PLANNING_ENABLED = getattr(config, "SHOT_PLANNING_ENABLED", True)
SHOT_PLANNING_MAX     = getattr(config, "SHOT_PLANNING_MAX", 4)

# Minimum duration for a single shot (seconds)
_MIN_SHOT_DURATION = 1.8


# ── Shot type definitions ─────────────────────────────────────────────────────

SHOT_TYPES = {
    "wide":     "wide establishing shot showing full scene and environment",
    "close":    "close-up shot showing emotion, expression, or key object",
    "reaction": "reaction shot showing character's emotional response",
    "detail":   "detail/insert shot showing a specific significant object or action",
}

# Beat → shot sequence to use
_BEAT_SHOTS = {
    "hook":       ["wide", "close"],
    "build_up":   ["wide", "close"],
    "discovery":  ["wide", "close", "reaction"],
    "climax":     ["wide", "close", "reaction", "detail"],
    "resolution": ["close", "wide"],
}

# Emotion modifiers for shot queries
_EMOTION_SHOT_STYLE = {
    "shock":      "sudden dramatic reveal, wide eyes, frozen moment",
    "suspense":   "slow tense reveal, dark lighting, nervous body language",
    "sadness":    "quiet still moment, soft light, downward gaze",
    "emotional":  "intimate frame, warm light, teary expression",
    "dramatic":   "high contrast lighting, intense expression, confrontation",
    "uplifting":  "bright environment, open body language, celebratory",
    "mysterious": "obscured face, dark background, slow movement",
    "funny":      "candid reaction, bright light, surprised expression",
}


def _has_groq():
    return (
        bool(config.GROQ_API_KEY)
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _groq_client():
    from groq import Groq
    return Groq(api_key=config.GROQ_API_KEY)


def _shots_for_beat(beat: str, emotion: str) -> list[str]:
    """Return the shot type sequence for this beat, capped by SHOT_PLANNING_MAX."""
    shots = _BEAT_SHOTS.get(beat, ["wide", "close"])
    return shots[:SHOT_PLANNING_MAX]


def _split_duration(total: float, n_shots: int) -> list[float]:
    """
    Split a scene's total duration across n shots.
    First and last shots get slightly more time (15% bonus each).
    """
    if n_shots <= 1:
        return [total]

    base = total / n_shots
    durations = [max(_MIN_SHOT_DURATION, base)] * n_shots

    # Boost first shot (establishing) and last shot (emotional beat)
    bonus = base * 0.15
    durations[0] = max(_MIN_SHOT_DURATION, durations[0] + bonus)
    durations[-1] = max(_MIN_SHOT_DURATION, durations[-1] + bonus)

    # Trim middle shots to compensate
    extra = (durations[0] - base) + (durations[-1] - base)
    if n_shots > 2:
        reduction_per_middle = extra / (n_shots - 2)
        for i in range(1, n_shots - 1):
            durations[i] = max(_MIN_SHOT_DURATION, durations[i] - reduction_per_middle)

    # Correct for rounding drift
    drift = total - sum(durations)
    durations[-1] = max(_MIN_SHOT_DURATION, durations[-1] + drift)
    return durations


def _query_for_shot(shot_type: str, scene: dict) -> list[str]:
    """Generate search queries for a specific shot type within a scene."""
    narration = scene.get("narration", "")
    emotion   = scene.get("emotion", "dramatic")
    beat      = scene.get("beat", "")
    style     = _EMOTION_SHOT_STYLE.get(emotion, "cinematic, expressive")

    base_subject = re.sub(
        r"show\s+", "", scene.get("visual_goal", "person in scene"), flags=re.IGNORECASE
    ).strip()

    queries = {
        "wide": [
            f"wide shot {base_subject}",
            f"establishing shot {base_subject} environment",
            f"full body {base_subject} room",
        ],
        "close": [
            f"close up face {style.split(',')[0].strip()}",
            f"portrait shot {emotion} expression person",
            f"tight shot {base_subject} {emotion}",
        ],
        "reaction": [
            f"reaction shot {emotion} person close up",
            f"person {style.split(',')[0].strip()} reacting",
            f"facial expression {emotion} shock surprise",
        ],
        "detail": [
            f"close up hand object detail {beat}",
            f"insert shot significant object scene",
            f"macro detail {base_subject}",
        ],
    }

    shot_queries = queries.get(shot_type, queries["wide"])
    return [q[:70] for q in shot_queries[:3]]


def _ai_shots_for_scene(scene: dict, n_shots: int) -> list[dict]:
    """
    Use Groq to generate shot-specific queries for each shot in the sequence.
    Returns a list of shot dicts. Falls back to rule-based on error.
    """
    if not _has_groq():
        return _rule_based_shots(scene, n_shots)

    try:
        client = _groq_client()
        shot_types = _shots_for_beat(scene.get("beat", "build_up"), scene.get("emotion", "dramatic"))[:n_shots]

        prompt = f"""You are a video editor planning shots for a single scene in a YouTube Short.

Scene:
  Narration: "{scene.get("narration", "")}"
  Visual goal: "{scene.get("visual_goal", "")}"
  Emotion: {scene.get("emotion", "dramatic")}
  Beat: {scene.get("beat", "build_up")}

Shot sequence to fill: {shot_types}

For each shot type, write 3 concrete stock search queries.
Rules:
- wide shot = full environment, people visible, context established
- close-up = face, hands, or key object in tight frame
- reaction = someone visibly reacting emotionally
- detail/insert = single specific object or action in macro/tight frame

Each query: max 6 words, describe what a camera would physically capture.

Return ONLY this JSON:
{{
  "shots": [
    {{"type": "wide", "queries": ["q1", "q2", "q3"]}},
    {{"type": "close", "queries": ["q1", "q2", "q3"]}},
    ...
  ]
}}
"""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
        start = raw.find("{")
        end = raw.rfind("}")
        parsed = json.loads(raw[start:end + 1])
        shots = parsed.get("shots", [])
        if len(shots) >= 2:
            return shots
        return _rule_based_shots(scene, n_shots)

    except Exception as exc:
        print(f"    shot planner AI failed ({exc}), using rule-based")
        return _rule_based_shots(scene, n_shots)


def _rule_based_shots(scene: dict, n_shots: int) -> list[dict]:
    """Rule-based fallback shot generation."""
    shot_types = _shots_for_beat(scene.get("beat", "build_up"), scene.get("emotion", "dramatic"))[:n_shots]
    return [
        {"type": st, "queries": _query_for_shot(st, scene)}
        for st in shot_types
    ]


def expand_scene(scene: dict) -> list[dict]:
    """
    Expand a single scene dict into multiple shot-aware sub-scenes.

    Each sub-scene is a full scene dict with:
    - shot_type: the type of shot (wide/close/reaction/detail)
    - shot_index: position within the parent scene
    - parent_scene_id: links back to the original scene
    - duration: subdivided from the parent's duration
    - queries: shot-type-specific search queries
    """
    if not SHOT_PLANNING_ENABLED:
        return [scene]

    beat    = scene.get("beat", "build_up")
    emotion = scene.get("emotion", "dramatic")
    total_duration = float(scene.get("duration", 4.0))

    shot_types = _shots_for_beat(beat, emotion)
    n_shots    = len(shot_types)

    # Very short scenes (< 3s): don't split
    if total_duration < _MIN_SHOT_DURATION * 2 or n_shots <= 1:
        return [{**scene, "shot_type": "wide", "shot_index": 0, "parent_scene_id": scene.get("scene_id")}]

    # Get shot-specific queries
    shots_data = _ai_shots_for_scene(scene, n_shots)
    durations  = _split_duration(total_duration, n_shots)

    expanded = []
    for i, (shot_type, duration) in enumerate(zip(shot_types, durations)):
        # Get queries for this specific shot
        shot_queries = []
        for s in shots_data:
            if s.get("type") == shot_type:
                shot_queries = [str(q).strip() for q in s.get("queries", []) if str(q).strip()]
                break
        if not shot_queries:
            shot_queries = _query_for_shot(shot_type, scene)

        # Transition: only last shot of each scene gets the scene's transition
        # mid-scene shots use quick cuts to maintain rhythm
        transition = scene.get("transition", "hard_cut") if i == n_shots - 1 else "hard_cut"

        sub_scene = {
            **scene,
            "scene_id":        f"{scene.get('scene_id', 1)}.{i + 1}",
            "parent_scene_id": scene.get("scene_id", 1),
            "shot_type":       shot_type,
            "shot_index":      i,
            "shot_total":      n_shots,
            "duration":        round(duration, 1),
            "queries":         shot_queries[:4],
            "transition":      transition,
            # Preferred asset type per shot type
            "asset_types":     _shot_asset_types(shot_type, scene),
        }
        expanded.append(sub_scene)

    return expanded


def _shot_asset_types(shot_type: str, scene: dict) -> list[str]:
    """Which asset type is preferred for each shot type."""
    # Action/reaction shots benefit from video; establishing and detail from images
    if shot_type in ("reaction",):
        return ["video", "image"]
    if shot_type in ("close", "detail"):
        return ["image", "video"]
    # wide: use the scene's original preference
    return scene.get("asset_types", ["image", "video"])


def expand_scene_plan(scene_plan: list) -> list:
    """
    Expand every scene in the plan into multi-shot sub-scenes.

    The total duration is preserved: it's distributed across the shots,
    not multiplied. The result has more scenes (assets needed) but the
    same overall video length.
    """
    if not SHOT_PLANNING_ENABLED:
        return scene_plan

    expanded = []
    for scene in scene_plan:
        shots = expand_scene(scene)
        expanded.extend(shots)
        if len(shots) > 1:
            shot_names = [s["shot_type"] for s in shots]
            print(
                f"  📷 Scene {scene.get('scene_id')} ({scene.get('beat')}) → "
                f"{len(shots)} shots: {' + '.join(shot_names)}"
            )

    print(f"  📷 Shot planning: {len(scene_plan)} scenes → {len(expanded)} shots")
    return expanded