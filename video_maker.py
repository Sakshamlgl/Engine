import os
import random
import shutil

import PIL
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)

import config
from subtitle_maker import make_subtitle_clips
from scene_builder  import build_scene_sequence

OUT_W = config.VIDEO_WIDTH    # 1080
OUT_H = config.VIDEO_HEIGHT   # 1920


# ── ImageMagick: auto-detect instead of hardcoding ────────

def _configure_imagemagick():
    """Find ImageMagick automatically on any OS — no hardcoded paths."""
    import moviepy.config as mpconf

    # If already set and valid, leave it alone
    current = getattr(mpconf, "IMAGEMAGICK_BINARY", None)
    if current and os.path.isfile(str(current)):
        return

    candidates = []

    # Windows: common install locations
    if os.name == "nt":
        program_files = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
            os.environ.get("LocalAppData", ""),
        ]
        for base in program_files:
            if not base:
                continue
            try:
                for entry in os.scandir(base):
                    if "ImageMagick" in entry.name and entry.is_dir():
                        for exe in ("magick.exe", "convert.exe"):
                            path = os.path.join(entry.path, exe)
                            if os.path.isfile(path):
                                candidates.append(path)
            except PermissionError:
                pass
        # Also try PATH
        for exe in ("magick.exe", "convert.exe"):
            found = shutil.which(exe)
            if found:
                candidates.append(found)
    else:
        # Linux / macOS: just check PATH
        for exe in ("magick", "convert"):
            found = shutil.which(exe)
            if found:
                candidates.append(found)

    if candidates:
        mpconf.IMAGEMAGICK_BINARY = candidates[0]
        print(f"  ImageMagick: {candidates[0]}")
    else:
        print("  ⚠️  ImageMagick not found — text subtitles may be unavailable.")
        print("      Install from https://imagemagick.org/script/download.php")


_configure_imagemagick()


# ── Helpers ────────────────────────────────────────────────

def _make_vertical(clip):
    """Convert landscape footage to 9:16 vertical (1080×1920)."""
    clip = clip.resize(height=OUT_H)
    if clip.w >= OUT_W:
        x_center = clip.w / 2
        clip = clip.crop(x1=x_center - OUT_W / 2, x2=x_center + OUT_W / 2)
    else:
        bg    = ColorClip(size=(OUT_W, OUT_H), color=(0, 0, 0), duration=clip.duration)
        x_off = (OUT_W - clip.w) // 2
        clip  = CompositeVideoClip([bg, clip.set_position((x_off, 0))])
    return clip


def _loop_video_to(clip, duration):
    """Loop a video clip until it reaches at least `duration` seconds."""
    if clip.duration >= duration:
        return clip
    loops = int(duration / clip.duration) + 1
    return concatenate_videoclips([clip] * loops).subclip(0, duration)


def _loop_audio_to(clip, duration):
    """Loop an AudioFileClip until it reaches at least `duration` seconds."""
    if clip.duration >= duration:
        return clip.subclip(0, duration)
    loops = int(duration / clip.duration) + 1
    return concatenate_audioclips([clip] * loops).subclip(0, duration)


def _mix_audio(voiceover, music_path, duration):
    """Mix voiceover + background music at 12% volume."""
    music = AudioFileClip(music_path)
    music = _loop_audio_to(music, duration)   # ← was _loop_video_to (bug: wrong type)
    music = music.volumex(0.12)
    return CompositeAudioClip([voiceover, music])


# ── Legacy fallback: single background video ──────────────

def _build_background_from_single(bg_video, duration):
    """
    Fallback for when no scene_assets are provided.
    Loops a single background video to fill the duration.
    """
    gameplay = VideoFileClip(bg_video, audio=False)
    gameplay = _loop_video_to(gameplay, duration + 1)

    max_start = max(0, gameplay.duration - duration - 0.5)
    start_at  = random.uniform(0, max_start) if max_start > 0 else 0
    gameplay  = gameplay.subclip(start_at, start_at + duration)
    return _make_vertical(gameplay)


# ── Main entry point ───────────────────────────────────────

def create_video(
    audio_path,
    output_path,
    script=None,
    music_path=None,
    scene_assets=None,   # List of scene dicts from fetch_assets()
    bg_video=None,       # Legacy single-background fallback
):
    """
    Assemble the final 9:16 Short.

    If scene_assets is provided, uses the scene_builder pipeline
    (Phase 8–11: Ken Burns, transitions, emotion-aware editing).

    Falls back to a single looped background video if scene_assets
    is empty or None.
    """
    visual = None

    print("  Loading audio...")
    audio    = AudioFileClip(audio_path)
    duration = audio.duration
    print(f"  Duration: {duration:.1f}s {'✅' if duration <= 60 else '⚠️ over 60s'}")

    # ── Build visual background ────────────────────────────
    if scene_assets:
        print(f"  Building scene sequence ({len(scene_assets)} scenes)...")
        try:
            visual = build_scene_sequence(scene_assets, target_duration=duration)
            print("  ✅ Scene sequence built")
        except Exception as e:
            print(f"  ❌ Scene sequence failed: {e}")
            audio.close()
            return None
    else:
        print("  ❌ No scene assets provided. Aborting.")
        audio.close()
        return None

    # ── Audio mix ──────────────────────────────────────────
    if music_path and os.path.exists(music_path):
        print("  Mixing voiceover + background music...")
        try:
            final_audio = _mix_audio(audio, music_path, duration)
        except Exception as e:
            print(f"  ⚠️ Music mix failed ({e}), voiceover only")
            final_audio = audio
    else:
        final_audio = audio

    # ── Subtitles ──────────────────────────────────────────
    layers = [visual]
    if script:
        print("  Generating subtitles...")
        try:
            sub_clips = make_subtitle_clips(script, duration, video_size=(OUT_W, OUT_H))
            layers   += sub_clips
            print(f"  ✅ {len(sub_clips)} subtitle clips created")
        except Exception as e:
            print(f"  ⚠️ Subtitles failed ({e}), skipping")

    # ── Render ─────────────────────────────────────────────
    final = CompositeVideoClip(layers, size=(OUT_W, OUT_H)).set_audio(final_audio)
    print(f"  Rendering {OUT_W}×{OUT_H} Short ({duration:.0f}s)...")

    final.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    audio.close()
    visual.close()
    final.close()

    print(f"  ✅ Short saved → {output_path}")
    return output_path
