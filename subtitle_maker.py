"""
Subtitle Maker — Phase 9 polish

Two rendering paths selected automatically at runtime:

  Path A (ImageMagick available):
    MoviePy TextClip — high-quality font rendering, word spacing
    First word is gold, rest are white

  Path B (PIL fallback — no ImageMagick needed):
    PIL RGBA text rendered to a numpy mask and composited
    Works identically on any machine; font quality slightly lower

Timing is proportional to word count so every word gets equal screen time.
"""

import os
import platform
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip

# ── Constants ────────────────────────────────────────────────────────────────
WORDS_PER_CHUNK = 3
FONT_SIZE       = 88
Y_POSITION      = 0.72    # 72% down the frame
HIGHLIGHT       = (255, 215, 0, 255)   # gold — first word
TEXT_COLOR      = (255, 255, 255, 255)
STROKE_COLOR    = (0,   0,   0,   255)
LINE_H          = 160     # height of subtitle strip in pixels


# ── ImageMagick detection ─────────────────────────────────────────────────────

def _imagemagick_ok() -> bool:
    """Return True if ImageMagick is on PATH (TextClip will work)."""
    try:
        import moviepy.config as mpconf
        b = getattr(mpconf, "IMAGEMAGICK_BINARY", "")
        if b and os.path.isfile(b):
            return True
    except Exception:
        pass
    return any(shutil.which(n) for n in ("magick", "convert"))


_USE_IMAGEMAGICK = _imagemagick_ok()


# ── Timing ───────────────────────────────────────────────────────────────────

def _chunks(script: str) -> list[str]:
    words  = script.split()
    return [" ".join(words[i:i + WORDS_PER_CHUNK])
            for i in range(0, len(words), WORDS_PER_CHUNK)]


def _timings(chunks: list[str], total: float) -> list[tuple]:
    """Word-count proportional timing — each word gets equal screen time."""
    if not chunks:
        return []
    wc      = [max(1, len(c.split())) for c in chunks]
    tw      = max(1, sum(wc))
    cursor  = 0.0
    result  = []
    for chunk, w in zip(chunks, wc):
        dur    = max(0.30, (w / tw) * total)
        result.append((chunk, cursor, cursor + dur))
        cursor += dur
    # Re-scale to fill exactly total_duration
    scale   = total / max(result[-1][2], 0.001)
    return [(c, s * scale, e * scale) for c, s, e in result]


# ── PIL font loader ───────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    sys = platform.system()
    candidates = {
        "Windows": [r"C:\Windows\Fonts\arialbd.ttf",
                    r"C:\Windows\Fonts\calibrib.ttf"],
        "Darwin":  ["/Library/Fonts/Arial Bold.ttf",
                    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                    "/System/Library/Fonts/Helvetica.ttc"],
    }.get(sys, [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ])
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── PIL subtitle clip ─────────────────────────────────────────────────────────

def _pil_clip(chunk: str, start: float, duration: float, W: int, H: int):
    """Render one subtitle chunk as a PIL-based transparent overlay clip."""
    font   = _load_font(FONT_SIZE)
    words  = chunk.split()
    img    = Image.new("RGBA", (W, LINE_H), (0, 0, 0, 0))
    draw   = ImageDraw.Draw(img)

    # Measure word widths
    GAP   = 18
    bboxes = [draw.textbbox((0, 0), w + " ", font=font) for w in words]
    widths = [b[2] - b[0] for b in bboxes]
    total_w = sum(widths) + GAP * (len(words) - 1)
    x = max(0, (W - total_w) // 2)
    y = (LINE_H - (bboxes[0][3] - bboxes[0][1])) // 2

    for i, (word, ww) in enumerate(zip(words, widths)):
        color = HIGHLIGHT if i == 0 else TEXT_COLOR
        # Stroke
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                if dx or dy:
                    draw.text((x + dx, y + dy), word, font=font, fill=STROKE_COLOR)
        draw.text((x, y), word, font=font, fill=color)
        x += ww + GAP

    arr  = np.array(img)
    rgb  = arr[:, :, :3]
    mask = arr[:, :, 3].astype("float64") / 255.0

    y_pos = int(H * Y_POSITION)
    rc    = ImageClip(rgb,  duration=duration).set_start(start)
    mc    = ImageClip(mask, ismask=True, duration=duration).set_start(start)
    return rc.set_mask(mc).set_position(("center", y_pos))


# ── ImageMagick subtitle clip ─────────────────────────────────────────────────

def _textclip(chunk: str, start: float, duration: float, W: int, H: int):
    """Render one subtitle chunk with MoviePy TextClip (requires ImageMagick)."""
    from moviepy.editor import TextClip, CompositeVideoClip, ColorClip

    words = chunk.split()
    clips = []
    GAP   = 20
    FONT  = "Arial-Bold"

    # Measure total width using single clip
    probe = TextClip(chunk, fontsize=FONT_SIZE, font=FONT,
                     color="white", method="label")
    total_w = probe.w
    probe.close()

    x = max(0, (W - total_w) // 2)
    y = int(H * Y_POSITION)

    for i, word in enumerate(words):
        color = "#FFD700" if i == 0 else "white"
        wc = (TextClip(word, fontsize=FONT_SIZE, font=FONT,
                       color=color, stroke_color="black", stroke_width=4,
                       method="label")
              .set_duration(duration).set_start(start))
        clips.append(wc.set_position((x, y)))
        x += wc.w + GAP

    return clips


# ── Public API ────────────────────────────────────────────────────────────────

def make_subtitle_clips(script: str, total_duration: float,
                        video_size: tuple = (1080, 1920)) -> list:
    """
    Return a list of positioned clips that can be added to CompositeVideoClip.

    Automatically uses PIL if ImageMagick is not installed.
    """
    W, H   = video_size
    all_clips = []

    if not _USE_IMAGEMAGICK:
        print("  ℹ️  No ImageMagick — using PIL subtitle renderer")

    for (chunk, start, end) in _timings(_chunks(script), total_duration):
        dur = end - start
        if _USE_IMAGEMAGICK:
            try:
                all_clips.extend(_textclip(chunk, start, dur, W, H))
                continue
            except Exception:
                pass   # fall through to PIL
        all_clips.append(_pil_clip(chunk, start, dur, W, H))

    return all_clips
