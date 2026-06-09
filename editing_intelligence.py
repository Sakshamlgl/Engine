"""
Editing Intelligence (v5) — Beat-Aware Cut Decisions

Previous system:
    Scene → Shot → Shot → Shot
    (all rule-based, no decisions)

New system:
    Story Brain + Beat → Editing Intelligence → Shot Decisions
    - Use reaction shot first?
    - Cut faster here?
    - Pause here?
    - Zoom harder?
    - Ken Burns direction?

This module analyses the scene plan and annotates every shot with
production-level editing directives that video_maker.py respects.

Key decisions made per shot:
    cut_style     : "hard" | "soft" | "smash"   — how to enter this shot
    pacing        : "slow" | "normal" | "fast"  — beat duration multiplier
    ken_burns     : {"direction": "in"|"out"|"left"|"right", "intensity": 1..3}
    hold_frames   : int — extra still frames at start (0 for action, 12 for reveal)
    audio_duck    : bool — duck music for this shot (big dialogue moment)
    reaction_first: bool — swap shot order (reaction before action)

Integration:
    scene_planner or main.py calls annotate_edit_decisions(expanded_plan)
    video_maker.py reads shot["edit"] for these directives
"""

from __future__ import annotations

import json
import re

import config


# ── Beat → editing profile ────────────────────────────────────────────────────

_BEAT_PROFILES: dict[str, dict] = {
    "hook": {
        "pacing":         "fast",
        "cut_style":      "hard",
        "ken_direction":  "in",
        "ken_intensity":  2,
        "hold_frames":    4,
        "audio_duck":     False,
        "reaction_first": False,
    },
    "build_up": {
        "pacing":         "normal",
        "cut_style":      "soft",
        "ken_direction":  "in",
        "ken_intensity":  1,
        "hold_frames":    6,
        "audio_duck":     False,
        "reaction_first": False,
    },
    "discovery": {
        "pacing":         "slow",       # slow build to the reveal
        "cut_style":      "soft",
        "ken_direction":  "out",        # pull back to reveal context
        "ken_intensity":  2,
        "hold_frames":    12,           # pause on discovery
        "audio_duck":     True,         # let narration breathe
        "reaction_first": True,         # reaction THEN the thing reacted to = suspense
    },
    "climax": {
        "pacing":         "fast",
        "cut_style":      "smash",      # smash cut into climax
        "ken_direction":  "in",
        "ken_intensity":  3,            # aggressive zoom
        "hold_frames":    0,
        "audio_duck":     False,
        "reaction_first": False,
    },
    "resolution": {
        "pacing":         "slow",
        "cut_style":      "soft",
        "ken_direction":  "out",        # pull back = letting go
        "ken_intensity":  1,
        "hold_frames":    8,
        "audio_duck":     True,
        "reaction_first": False,
    },
}

# ── Emotion overrides ─────────────────────────────────────────────────────────

_EMOTION_OVERRIDES: dict[str, dict] = {
    "shock": {
        "cut_style":   "smash",
        "pacing":      "fast",
        "ken_intensity": 3,
        "hold_frames": 2,
    },
    "suspense": {
        "cut_style":   "soft",
        "pacing":      "slow",
        "ken_direction": "in",    # slow creeping zoom = tension
        "hold_frames": 14,        # uncomfortable hold
        "reaction_first": True,
    },
    "sadness": {
        "cut_style":   "soft",
        "pacing":      "slow",
        "ken_direction": "out",   # pulling away = distance/loss
        "hold_frames": 10,
        "audio_duck":  True,
    },
    "uplifting": {
        "pacing":      "normal",
        "cut_style":   "soft",
        "ken_direction": "out",   # expanding = hope
        "ken_intensity": 1,
    },
    "funny": {
        "pacing":      "fast",
        "cut_style":   "hard",
        "hold_frames": 0,
    },
}

# ── Shot type → pacing modifier ──────────────────────────────────────────────

_SHOT_TYPE_PACING: dict[str, float] = {
    "wide":     1.0,
    "close":    1.1,    # close-ups linger slightly longer
    "reaction": 0.85,   # reactions are punchy
    "detail":   0.9,
}


# ── LLM editing decisions (optional, higher quality) ─────────────────────────

def _has_groq():
    return (
        bool(config.GROQ_API_KEY)
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _llm_edit_decisions(scene: dict, shot: dict, arc_context: str) -> dict | None:
    """
    Ask the LLM to make nuanced editing decisions for a specific shot.
    Returns a partial override dict or None on failure.
    Only called for climax and key discovery beats (to save tokens).
    """
    if not _has_groq():
        return None

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        prompt = f"""You are a professional YouTube Shorts video editor.

Story arc context: {arc_context}

Current shot:
  Scene beat: {scene.get("beat")}
  Emotion: {scene.get("emotion")}
  Narration: "{scene.get("narration", "")[:100]}"
  Shot type: {shot.get("shot_type", "unknown")}

Decide the editing style for this ONE shot. Be specific and decisive.

Return ONLY this JSON:
{{
  "cut_style": "hard"|"soft"|"smash",
  "pacing": "slow"|"normal"|"fast",
  "ken_direction": "in"|"out"|"left"|"right",
  "ken_intensity": 1|2|3,
  "hold_frames": <0-20>,
  "reaction_first": true|false,
  "reason": "<one sentence>"
}}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
        )
        raw   = response.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            return None
        result = json.loads(raw[start:end + 1])
        reason = result.pop("reason", "")
        if reason:
            print(f"      ✂️  edit decision: {reason}")
        return result

    except Exception:
        return None


# ── Core annotation logic ─────────────────────────────────────────────────────

def _build_edit_profile(scene: dict, shot: dict) -> dict:
    """
    Build an editing profile for one shot by merging:
      1. Beat profile (baseline)
      2. Emotion overrides
      3. Shot-type pacing modifier
    """
    beat    = scene.get("beat", "build_up")
    emotion = scene.get("emotion", "dramatic")

    # Start with beat baseline
    profile = dict(_BEAT_PROFILES.get(beat, _BEAT_PROFILES["build_up"]))

    # Apply emotion overrides (partial merge — only keys that exist in override)
    for k, v in _EMOTION_OVERRIDES.get(emotion, {}).items():
        profile[k] = v

    # Shot-type pacing multiplier — store for video_maker
    shot_type     = shot.get("shot_type", "wide")
    pacing_factor = _SHOT_TYPE_PACING.get(shot_type, 1.0)
    profile["pacing_factor"] = pacing_factor

    return profile


def _arc_summary(scene_plan: list, current_idx: int) -> str:
    """Build a compact story-so-far for LLM context."""
    recent_start = max(0, current_idx - 3)
    parts = []
    for s in scene_plan[recent_start:current_idx + 1]:
        parts.append(f"[{s.get('beat', '?')}] {s.get('narration', '')[:50]}")
    return " → ".join(parts)


def annotate_edit_decisions(
    expanded_plan: list,
    use_llm_for_climax: bool = True,
) -> list:
    """
    Annotate every shot in the expanded plan with editing directives.

    Modifies each shot dict in-place, adding an "edit" key:
        shot["edit"] = {
            "cut_style":      str,
            "pacing":         str,
            "pacing_factor":  float,
            "ken_direction":  str,
            "ken_intensity":  int,
            "hold_frames":    int,
            "audio_duck":     bool,
            "reaction_first": bool,
        }

    For climax and discovery beats, optionally uses the LLM for higher-quality
    nuanced decisions (controlled by use_llm_for_climax).

    Args:
        expanded_plan:      List of shot dicts from shot_planner.expand_scene_plan()
        use_llm_for_climax: Use LLM for climax/discovery beats if Groq available

    Returns:
        Same list, each shot enriched with "edit" key.
    """
    print(f"\n  ✂️  Editing Intelligence: annotating {len(expanded_plan)} shots...")

    for idx, shot in enumerate(expanded_plan):
        beat = shot.get("beat", "build_up")

        # Rule-based profile (always computed)
        profile = _build_edit_profile(shot, shot)

        # LLM override for high-stakes beats (saves tokens by only calling on climax/discovery)
        if use_llm_for_climax and beat in ("climax", "discovery") and _has_groq():
            arc_ctx  = _arc_summary(expanded_plan, idx)
            llm_edit = _llm_edit_decisions(shot, shot, arc_ctx)
            if llm_edit:
                profile.update({k: v for k, v in llm_edit.items() if k in profile})

        shot["edit"] = profile

    # Post-pass: enforce reaction_first shot reordering
    _apply_reaction_reordering(expanded_plan)

    slow  = sum(1 for s in expanded_plan if s.get("edit", {}).get("pacing") == "slow")
    fast  = sum(1 for s in expanded_plan if s.get("edit", {}).get("pacing") == "fast")
    smash = sum(1 for s in expanded_plan if s.get("edit", {}).get("cut_style") == "smash")
    print(f"  ✂️  Done: {slow} slow, {fast} fast, {smash} smash cuts")

    return expanded_plan


def _apply_reaction_reordering(shots: list) -> None:
    """
    For shots marked reaction_first=True, swap with the preceding shot
    if the preceding shot is a wide shot (wide → reaction → close becomes
    reaction → wide → close, creating more suspense).

    Only swaps within the same scene (same narration).
    """
    i = 1
    while i < len(shots):
        shot   = shots[i]
        prev   = shots[i - 1]
        edit   = shot.get("edit", {})

        # Only swap if this shot wants reaction_first AND previous is a wide
        if (
            edit.get("reaction_first")
            and shot.get("shot_type") == "reaction"
            and prev.get("shot_type") == "wide"
            and shot.get("narration") == prev.get("narration")   # same scene
        ):
            shots[i - 1], shots[i] = shots[i], shots[i - 1]
            i += 2   # skip the newly-placed shot
        else:
            i += 1


# ── video_maker integration helpers ──────────────────────────────────────────

def get_ken_burns_params(shot: dict) -> dict:
    """
    Extract Ken Burns parameters from a shot's edit directive.
    Returns a dict compatible with video_maker's Ken Burns implementation.
    """
    edit = shot.get("edit", {})
    direction = edit.get("ken_direction", "in")
    intensity = int(edit.get("ken_intensity", 1))   # 1=subtle, 2=medium, 3=strong

    # Map intensity to zoom_factor
    zoom_factors = {1: 1.05, 2: 1.12, 3: 1.20}
    zoom_factor  = zoom_factors.get(intensity, 1.08)

    # Map direction to start/end pan
    direction_map = {
        "in":    {"zoom_start": 1.0,         "zoom_end": zoom_factor, "pan": "center"},
        "out":   {"zoom_start": zoom_factor,  "zoom_end": 1.0,        "pan": "center"},
        "left":  {"zoom_start": 1.0,         "zoom_end": zoom_factor, "pan": "left"},
        "right": {"zoom_start": 1.0,         "zoom_end": zoom_factor, "pan": "right"},
    }

    params = direction_map.get(direction, direction_map["in"])
    params["hold_frames"] = int(edit.get("hold_frames", 0))
    return params


def get_transition(shot: dict) -> str:
    """
    Map cut_style to a transition name for video_maker.
    Returns "cut" | "crossfade" | "smash_cut"
    """
    cut_style = shot.get("edit", {}).get("cut_style", "hard")
    return {
        "hard":  "cut",
        "soft":  "crossfade",
        "smash": "smash_cut",
    }.get(cut_style, "cut")
