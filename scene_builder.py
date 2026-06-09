"""
Scene Builder — Phase 8/9/10/11: Ken Burns + Emotion-Aware + Transitions

Builds the visual background track from per-scene assets.

Ken Burns:      emotion-driven zoom amount and pan pattern
Transitions:    hard_cut | crossfade | fade (through black) | push (slide)
Emotion-Aware:  zoom speed and transition type set by scene_planner
"""

import random

import numpy as np
from PIL import Image
from moviepy.editor import (
    ColorClip,
    CompositeVideoClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)

import config

OUT_W = config.VIDEO_WIDTH    # 1080
OUT_H = config.VIDEO_HEIGHT   # 1920

# ── Emotion → Ken Burns zoom intensity ───────────────────────────────────────
_EMOTION_ZOOM = {
    "shock":      0.16,
    "dramatic":   0.12,
    "suspense":   0.10,
    "mysterious": 0.10,
    "tense":      0.10,
    "emotional":  0.06,
    "sadness":    0.05,
    "uplifting":  0.08,
    "funny":      0.07,
}
_DEFAULT_ZOOM = 0.08


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_vertical_video(clip):
    """Resize + centre-crop any VideoClip to OUT_W × OUT_H."""
    clip = clip.resize(height=OUT_H)
    if clip.w >= OUT_W:
        xc   = clip.w / 2
        return clip.crop(x1=xc - OUT_W / 2, x2=xc + OUT_W / 2)
    bg   = ColorClip(size=(OUT_W, OUT_H), color=(0,0,0), duration=clip.duration)
    xoff = (OUT_W - clip.w) // 2
    return CompositeVideoClip([bg, clip.set_position((xoff, 0))], size=(OUT_W, OUT_H))


def _pan_points(index, max_x, max_y):
    """Four distinct pan directions, chosen by scene index."""
    patterns = [
        ((0, 0),         (max_x, max_y)),
        ((max_x, 0),     (0, max_y)),
        ((max_x//2, 0),  (max_x//2, max_y)),
        ((0, max_y//2),  (max_x, max_y//2)),
    ]
    return patterns[index % len(patterns)]


# ── Clip builders ─────────────────────────────────────────────────────────────

def build_image_clip(path: str, duration: float, emotion: str = "dramatic", scene_id: int = 1):
    """
    Still image → Ken Burns documentary clip.
    Emotion controls zoom amount; scene_id controls pan direction.
    """
    source = Image.open(path).convert("RGB")
    sw, sh = source.size

    # Crop to 9:16 ratio at source resolution
    target_ratio = OUT_W / OUT_H
    source_ratio = sw / max(1, sh)
    if source_ratio > target_ratio:
        crop_h = sh
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = sw
        crop_h = int(crop_w / target_ratio)
    crop_w = max(1, min(crop_w, sw))
    crop_h = max(1, min(crop_h, sh))

    max_x = max(0, sw - crop_w)
    max_y = max(0, sh - crop_h)
    start_pt, end_pt = _pan_points(scene_id, max_x, max_y)

    zoom_amount = _EMOTION_ZOOM.get(str(emotion).lower(), _DEFAULT_ZOOM)

    def make_frame(t: float) -> np.ndarray:
        progress = min(1.0, max(0.0, t / max(duration, 0.001)))
        zoom     = 1.0 + zoom_amount * progress
        cur_w    = max(1, int(crop_w / zoom))
        cur_h    = max(1, int(crop_h / zoom))

        x = int(start_pt[0] * (1-progress) + end_pt[0] * progress)
        y = int(start_pt[1] * (1-progress) + end_pt[1] * progress)
        # Centre the zoom crop
        x += int((crop_w - cur_w) / 2)
        y += int((crop_h - cur_h) / 2)
        x  = max(0, min(sw - cur_w, x))
        y  = max(0, min(sh - cur_h, y))

        frame = source.crop((x, y, x + cur_w, y + cur_h))
        frame = frame.resize((OUT_W, OUT_H), Image.LANCZOS)
        return np.array(frame)

    clip     = VideoClip(make_frame=make_frame, duration=duration)
    clip.fps = 30          # required by MoviePy for rendering
    return clip


def build_video_clip(path: str, duration: float):
    """Load video, loop if short, random-start subclip, crop to 9:16."""
    clip = VideoFileClip(path, audio=False)
    if clip.duration < duration:
        loops = int(duration / max(clip.duration, 0.1)) + 2
        clip  = concatenate_videoclips([clip] * loops)
    max_start = max(0, clip.duration - duration - 0.25)
    start     = random.uniform(0, max_start) if max_start > 0 else 0
    clip      = clip.subclip(start, start + duration)
    return make_vertical_video(clip.set_duration(duration))


# ── Transition engine (Phase 11) ──────────────────────────────────────────────

def _transition_secs(asset: dict) -> float:
    """How many seconds of overlap/fade for this scene's transition."""
    t  = str(asset.get("transition", "")).lower()
    em = str(asset.get("emotion", "")).lower()
    if t == "hard_cut" or em == "shock":
        return 0.0
    if t == "fade" or em in ("sadness", "emotional"):
        return 0.50
    if t == "push":
        return 0.35
    if t == "crossfade" or em in ("suspense", "mysterious"):
        return 0.35
    return 0.20     # default soft cut


def _crossfade(clip1, clip2, t: float):
    """Opacity blend — clip1 fades out while clip2 fades in."""
    try:
        c1  = clip1.crossfadeout(t)
        c2  = clip2.set_start(clip1.duration - t).crossfadein(t)
        out = CompositeVideoClip([c1, c2], size=(OUT_W, OUT_H))
        return out.subclip(0, clip1.duration + clip2.duration - t)
    except Exception as e:
        print(f"    ⚠️  crossfade failed ({e}), using hard cut")
        return concatenate_videoclips([clip1, clip2], method="compose")


def _fade(clip1, clip2, t: float):
    """True fade: clip1 → black → clip2."""
    try:
        c1 = clip1.fadeout(t)
        c2 = clip2.fadein(t)
        return concatenate_videoclips([c1, c2], method="compose")
    except Exception as e:
        print(f"    ⚠️  fade failed ({e}), using hard cut")
        return concatenate_videoclips([clip1, clip2], method="compose")


def _push(clip1, clip2, t: float):
    """
    Horizontal push: clip2 slides in from the right while clip1 slides out left.
    Both clips are visible simultaneously during the transition.
    """
    try:
        # Transition segment only
        c1_end   = clip1.subclip(clip1.duration - t)
        c2_start = clip2.subclip(0, t)

        # Animate positions: t=0 → side by side; t=t → swapped
        c1_end   = c1_end.set_position(
            lambda s, dur=t: (-int(OUT_W * s / max(dur, 0.001)), 0)
        )
        c2_start = c2_start.set_position(
            lambda s, dur=t: (OUT_W - int(OUT_W * s / max(dur, 0.001)), 0)
        )
        trans = CompositeVideoClip([c1_end, c2_start], size=(OUT_W, OUT_H)).set_duration(t)

        return concatenate_videoclips([
            clip1.subclip(0, clip1.duration - t),
            trans,
            clip2.subclip(t),
        ], method="compose")
    except Exception as e:
        print(f"    ⚠️  push failed ({e}), using crossfade")
        return _crossfade(clip1, clip2, t)


def _apply_transition(clip1, clip2, transition: str, t: float):
    if t <= 0 or transition == "hard_cut":
        return concatenate_videoclips([clip1, clip2], method="compose")
    if transition == "fade":
        return _fade(clip1, clip2, t)
    if transition == "push":
        return _push(clip1, clip2, t)
    return _crossfade(clip1, clip2, t)   # default: crossfade


# ── Duration scaling ──────────────────────────────────────────────────────────

def _scaled_durations(scene_assets: list, target_duration: float) -> list:
    """Scale scene durations proportionally to fill target_duration exactly."""
    requested = [max(1.5, float(a.get("duration") or 3)) for a in scene_assets]
    overlaps  = [_transition_secs(a) for a in scene_assets[1:]]
    # Total raw clip time needed: target + overlaps (because overlaps are shared time)
    visual_total = target_duration + sum(overlaps)
    scale = visual_total / max(1.0, sum(requested))
    scaled = [max(1.5, r * scale) for r in requested]
    # Fix rounding drift in last clip
    drift        = visual_total - sum(scaled)
    scaled[-1]   = max(1.5, scaled[-1] + drift)
    return scaled


# ── Main assembler ────────────────────────────────────────────────────────────

def build_scene_clip(asset: dict, duration: float):
    """Build the right clip type for this scene asset."""
    if asset.get("type") == "video":
        return build_video_clip(asset["asset"], duration)
    return build_image_clip(
        asset["asset"],
        duration,
        emotion  = asset.get("emotion", "dramatic"),
        scene_id = int(float(asset.get("scene_id") or 1)),
    )


def build_scene_sequence(scene_assets: list, target_duration: float):
    """
    Build the full visual background from per-scene assets.

    Applies per-transition effects between scenes and scales all clip
    durations so the total matches target_duration exactly.
    """
    if not scene_assets:
        raise ValueError("scene_assets is empty")

    durations = _scaled_durations(scene_assets, target_duration)
    clips     = []

    for i, (asset, dur) in enumerate(zip(scene_assets, durations)):
        clip = build_scene_clip(asset, dur)
        clips.append(clip)
        print(f"    Scene {i+1}/{len(scene_assets)}: "
              f"{asset.get('type','img')} | {asset.get('emotion','?'):10s} | "
              f"{dur:.1f}s | {asset.get('transition','cut')}")

    # Merge clips with transitions
    result = clips[0]
    for i in range(1, len(clips)):
        transition = str(scene_assets[i].get("transition", "hard_cut")).lower()
        t_secs     = _transition_secs(scene_assets[i])
        result     = _apply_transition(result, clips[i], transition, t_secs)

    # Trim or pad to exact target
    if result.duration > target_duration:
        result = result.subclip(0, target_duration)
    elif result.duration < target_duration - 0.05:
        pad    = ColorClip(size=(OUT_W, OUT_H), color=(0,0,0),
                           duration=target_duration - result.duration).set_fps(30)
        result = concatenate_videoclips([result, pad], method="compose")

    return result.set_duration(target_duration)
