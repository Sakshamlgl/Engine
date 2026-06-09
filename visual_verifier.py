"""
Visual Verifier — Vision-Model Asset Checking

The current pipeline assumes: Search Result = Correct.
This module breaks that assumption by actually LOOKING at each downloaded
asset and asking a vision model whether it matches the scene narration.

Flow:
    Asset downloaded
         ↓
    visual_verify(scene, asset_path)
         ↓
    Vision model inspects the image/video thumbnail
         ↓
    Returns (passes: bool, score: int, reason: str)
         ↓
    asset_fetcher skips assets that fail verification

For videos, the first frame is extracted as a thumbnail for inspection.

Configuration (config.py / .env):
    VISUAL_VERIFY_ENABLED     = true   # master switch (default: true)
    VISUAL_VERIFY_THRESHOLD   = 35     # 0-100, below this = fail (default: 35)
    VISUAL_VERIFY_PROVIDER    = groq   # "groq" uses llama-3.2-11b-vision (free)

Note: Uses Groq's free vision model (llama-3.2-11b-vision-preview) so this
adds ~1-2s per scene with no additional cost.
"""

import base64
import io
import json
import os
import re
import tempfile

import config

VISUAL_VERIFY_ENABLED   = getattr(config, "VISUAL_VERIFY_ENABLED",   True)
VISUAL_VERIFY_THRESHOLD = getattr(config, "VISUAL_VERIFY_THRESHOLD",  35)


def _has_groq():
    return (
        bool(config.GROQ_API_KEY)
        and config.GROQ_API_KEY != "your_groq_api_key"
    )


def _extract_video_thumbnail(video_path: str) -> str | None:
    """Extract the first frame of a video as a JPEG temp file. Returns path or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, frame)
        return tmp.name
    except ImportError:
        # cv2 not available — try moviepy
        try:
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            frame_path = video_path + "_thumb.jpg"
            clip.save_frame(frame_path, t=min(0.5, clip.duration / 2))
            clip.close()
            return frame_path
        except Exception:
            return None
    except Exception:
        return None


def _encode_image_b64(image_path: str) -> tuple[str, str]:
    """
    Returns (base64_data, media_type) for an image file.
    Resizes large images to save tokens.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        # Resize to max 512px on longest side to save tokens
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
        # Raw file encode fallback
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        return data, media_type


def visual_verify(scene: dict, asset_path: str) -> tuple[bool, int, str]:
    """
    Ask a vision model if the asset at asset_path matches the scene.

    Args:
        scene:      Scene dict with narration, visual_goal, emotion
        asset_path: Path to the downloaded image or video file

    Returns:
        (passes, score, reason)
        passes: True if score >= VISUAL_VERIFY_THRESHOLD
        score:  0-100
        reason: one-sentence explanation
    """
    if not VISUAL_VERIFY_ENABLED:
        return True, 100, "verification disabled"

    if not os.path.exists(asset_path):
        return False, 0, "asset file not found"

    if not _has_groq():
        return True, 60, "groq unavailable, skipping vision check"

    # Determine image to send
    ext = os.path.splitext(asset_path)[1].lower()
    thumbnail_path = None

    if ext in (".mp4", ".mov", ".avi", ".webm"):
        thumbnail_path = _extract_video_thumbnail(asset_path)
        if not thumbnail_path:
            # Can't inspect video — give it a pass with moderate score
            return True, 55, "video thumbnail extraction failed, skipped"
        inspect_path = thumbnail_path
    else:
        inspect_path = asset_path

    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)

        image_data, media_type = _encode_image_b64(inspect_path)

        prompt = f"""You are a quality control editor for a YouTube Shorts video.

The video has this scene:
  Narration: "{scene.get("narration", "")}"
  Visual intent: "{scene.get("visual_goal", "")}"
  Required emotion: {scene.get("emotion", "")}

Look at this image (it is {'the first frame of a video clip' if ext in ('.mp4', '.mov', '.avi', '.webm') else 'a stock photo'}).

Rate: Does this image/clip make sense as a visual for the scene described above?
Score from 0 to 100:
  80-100 = strong visual match — clearly relevant
  60-79  = decent match — related and appropriate
  40-59  = partial — loosely related
  20-39  = weak — doesn't really fit the scene
  0-19   = wrong — misleading, confusing, or completely off

Return ONLY this JSON:
{{"score": <0-100>, "reason": "<one sentence describing what you see and why it matches or doesn't>"}}"""

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON in vision response")

        result = json.loads(raw[start:end + 1])
        score = max(0, min(100, int(result.get("score", 50))))
        reason = str(result.get("reason", "")).strip()
        passes = score >= VISUAL_VERIFY_THRESHOLD

        return passes, score, reason

    except Exception as exc:
        # Vision check failed — give a pass to avoid breaking the pipeline
        return True, 50, f"vision check error ({exc})"

    finally:
        # Clean up temporary thumbnail
        if thumbnail_path and os.path.exists(thumbnail_path):
            try:
                os.unlink(thumbnail_path)
            except Exception:
                pass


def log_verification(scene_index: int, asset: dict, passes: bool, score: int, reason: str) -> None:
    """Print a consistent verification log line."""
    status = "✅" if passes else "❌"
    asset_type = asset.get("type", "?")
    provider = asset.get("provider", "?")
    print(
        f"    👁  Vision verify scene {scene_index} "
        f"({asset_type}/{provider}): {status} score={score} — {reason}"
    )