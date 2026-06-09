"""
Image Generator — Phase 14/4: AI Image Fallback (Grok-first)

When stock search + visual verifier rejects an asset (low vision score),
this module generates a replacement image using strict tiered fallbacks:

  Tier 1 — Grok (xAI)          — "Super Grok" image generation via xAI API
  Tier 2 — Pollinations.ai     — model=turbo (or z-image-turbo) for speed/quality
  Tier 3 — Local PIL art       — instant geometric emotion-themed fallback

Requires XAI_API_KEY in .env for Tier 1.

Configure in .env:
  XAI_API_KEY=your_xai_key
  POLLINATIONS_ENABLED=true
  POLLINATIONS_TIMEOUT=25
"""

import hashlib
import math
import os
import random

import requests
from PIL import Image, ImageDraw, ImageFilter

import config

# ── Config ────────────────────────────────────────────────────────────────────

XAI_API_KEY = getattr(config, "XAI_API_KEY", os.getenv("XAI_API_KEY", ""))
POLLINATIONS_ENABLED = os.getenv("POLLINATIONS_ENABLED", "true").lower() != "false"
POLLINATIONS_TIMEOUT = int(os.getenv("POLLINATIONS_TIMEOUT", "25"))

# Phase 4: New Grok image generation timeout
GROK_IMAGE_TIMEOUT = int(os.getenv("GROK_IMAGE_TIMEOUT", "45"))

# ── Phase 4: Grok (xAI) image generation ──────────────────────────────────────
def _has_xai_key():
    return bool(XAI_API_KEY) and XAI_API_KEY != "your_xai_key" and len(XAI_API_KEY) > 10


def _generate_with_grok(scene: dict, output_path: str) -> str | None:
    """
    Tier 1: Generate image using xAI Grok image model.
    Endpoint follows the standard xAI / OpenAI-compatible images/generations.
    """
    if not _has_xai_key():
        return None

    prompt = _build_prompt(scene)
    # Enhance prompt for Grok strengths (photorealistic, cinematic, vertical)
    full_prompt = (
        f"{prompt}, highly detailed, cinematic lighting, 9:16 vertical composition, "
        "professional photography, no text, no watermark, no logos"
    )

    url = "https://api.x.ai/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "grok-2-image",   # or "grok-2" depending on current xAI offering; "Super Grok" style
        "prompt": full_prompt,
        "n": 1,
        "size": "1080x1920",       # closest supported; server may adjust
        "response_format": "url",
    }

    print(f'    🤖 Grok (xAI) generating for: "{scene.get("visual_goal", "")[:60]}..."')
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=GROK_IMAGE_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        image_url = None
        if "data" in data and data["data"]:
            image_url = data["data"][0].get("url")
        elif "images" in data and data["images"]:
            image_url = data["images"][0].get("url")  # alternative shape

        if not image_url:
            raise ValueError("No image URL returned from Grok")

        # Download the generated image
        img_resp = requests.get(image_url, timeout=30, headers={"User-Agent": "YTShortsBot/5.0"})
        img_resp.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(img_resp.content)

        kb = os.path.getsize(output_path) // 1024
        print(f"    ✅ Grok image ({kb}KB) → {os.path.basename(output_path)}")
        return output_path

    except Exception as e:
        print(f"    ⚠️  Grok image generation failed ({e})")
        return None


# ── Emotion → visual style ────────────────────────────────────────────────────

_STYLE_MAP = {
    "suspense":   "cinematic dark moody lighting, film noir, deep shadows, blue tones",
    "shock":      "dramatic high contrast, harsh studio lighting, vivid colors",
    "sadness":    "soft natural light, muted desaturated colors, melancholic atmosphere",
    "emotional":  "warm golden-hour light, intimate portrait, shallow depth of field",
    "dramatic":   "cinematic wide angle, dramatic side lighting, professional photography",
    "uplifting":  "bright airy natural light, vibrant colors, hopeful optimistic atmosphere",
    "mysterious": "foggy atmospheric, dim ambient blue light, mystery, eerie depth",
    "funny":      "bright cheerful lighting, candid casual, warm natural colors",
}
_DEFAULT_STYLE = "cinematic, professional photography, 4K ultra-detailed"

# ── PIL palette fallback ──────────────────────────────────────────────────────

_PALETTES = {
    "suspense":   ((12, 18, 34),  (58, 67, 91),   (185, 194, 213)),
    "shock":      ((42, 10, 16),  (110, 23, 38),   (235, 178, 133)),
    "sadness":    ((21, 30, 44),  (60, 82, 105),   (185, 203, 219)),
    "emotional":  ((31, 24, 42),  (86, 59, 96),    (226, 197, 198)),
    "uplifting":  ((16, 49, 64),  (49, 128, 128),  (225, 221, 160)),
    "funny":      ((40, 41, 25),  (126, 118, 44),  (242, 220, 104)),
    "mysterious": ((18, 17, 24),  (52, 49, 70),    (179, 170, 201)),
    "dramatic":   ((22, 19, 22),  (73, 62, 71),    (232, 196, 169)),
}


# ════════════════════════════════════════════════════════════════════════════
# Tier 1 — Pollinations.ai
# ════════════════════════════════════════════════════════════════════════════

def _build_prompt(scene: dict) -> str:
    goal    = scene.get("visual_goal", "cinematic scene")
    emotion = scene.get("emotion", "dramatic")
    style   = _STYLE_MAP.get(emotion, _DEFAULT_STYLE)
    beat    = scene.get("beat", "")

    # Extra guidance from beat
    beat_hint = {
        "hook":       "attention-grabbing, impactful opening",
        "climax":     "peak emotional intensity, dramatic reveal",
        "resolution": "sense of closure, aftermath",
    }.get(beat, "")

    parts = [
        goal,
        style,
        beat_hint,
        "portrait orientation 9:16",
        "no text, no watermark, no UI elements, photorealistic",
    ]
    return ", ".join(p for p in parts if p)


def _pollinations(scene: dict, output_path: str) -> str | None:
    """Request a photorealistic image from Pollinations.ai."""
    if not POLLINATIONS_ENABLED:
        return None

    prompt = _build_prompt(scene)
    # Deterministic seed per visual_goal so re-runs get the same image
    seed = int(hashlib.md5(scene.get("visual_goal", "").encode()).hexdigest()[:8], 16) % 99999

    encoded = requests.utils.quote(prompt, safe="")
    url     = f"https://image.pollinations.ai/prompt/{encoded}"
    # Phase 4: Prefer turbo / fast model for fallback (per plan)
    params  = {"width": 1080, "height": 1920, "nologo": "true",
                "model": "turbo", "seed": seed, "enhance": "true"}

    print(f'    🎨 Pollinations.ai → "{prompt[:70]}..."')
    try:
        r = requests.get(url, params=params, timeout=POLLINATIONS_TIMEOUT,
                         headers={"User-Agent": "YTShortsBot/3.0"})
        r.raise_for_status()

        # Validate we got image data (not an error HTML page)
        if len(r.content) < 5000 or not r.content[:4] in (b'\xff\xd8\xff', b'\x89PNG'):
            raise ValueError("Response is not a valid image")

        with open(output_path, "wb") as f:
            f.write(r.content)
        kb = os.path.getsize(output_path) // 1024
        print(f"    ✅ AI image ({kb}KB) → {os.path.basename(output_path)}")
        return output_path

    except Exception as e:
        print(f"    ⚠️  Pollinations.ai failed ({e})")
        return None


# ════════════════════════════════════════════════════════════════════════════
# Tier 2 — Local PIL artwork
# ════════════════════════════════════════════════════════════════════════════

def _palette(emotion: str):
    return _PALETTES.get(str(emotion).lower(), _PALETTES["dramatic"])


def _blend(a, b, t):
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))


def _gradient(size, colors):
    w, h = size
    top, mid, bot = colors
    img = Image.new("RGB", size, top)
    px  = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        c = _blend(top, mid, t / 0.55) if t < 0.55 else _blend(mid, bot, (t - 0.55) / 0.45)
        vig = 1 - min(0.42, abs(0 - 0) / w + abs(y - h / 2) / h)
        for x in range(w):
            vig_x = 1 - min(0.42, abs(x - w / 2) / w + abs(y - h / 2) / h)
            px[x, y] = tuple(max(0, min(255, int(ci * vig_x))) for ci in c)
    return img


def _draw_phone(draw, cx, cy, scale, accent):
    w, h, r = int(190*scale), int(360*scale), int(24*scale)
    draw.rounded_rectangle((cx-w//2, cy-h//2, cx+w//2, cy+h//2), r,
                            fill=(18,20,25), outline=accent, width=max(3, int(4*scale)))
    draw.rounded_rectangle((cx-w//2+18, cy-h//2+42, cx+w//2-18, cy+h//2-42),
                            12, fill=(220,228,238))
    for i in range(5):
        y = cy - int(90*scale) + i * int(42*scale)
        draw.rounded_rectangle((cx-int(55*scale), y, cx+int(55*scale), y+int(20*scale)),
                                8, fill=(115,132,158))


def _draw_people(draw, size, accent):
    w, h = size
    y = int(h * 0.63)
    for x, facing in ((int(w*0.38), 1), (int(w*0.62), -1)):
        draw.ellipse((x-48,y-210,x+48,y-114), fill=(22,22,27), outline=accent, width=4)
        draw.rounded_rectangle((x-70,y-110,x+70,y+120), 38, fill=(28,30,38))
        arm_x = x + facing * 95
        draw.line((x+facing*45, y-40, arm_x, y+35), fill=accent, width=10)


def _draw_science(draw, size, accent):
    w, h = size
    cx, cy = w//2, int(h*0.45)
    for r in (320, 230, 145):
        draw.ellipse((cx-r,cy-r,cx+r,cy+r), outline=accent, width=5)
    for i in range(20):
        ang = i * math.pi * 2 / 20
        px  = cx + int(math.cos(ang) * random.randint(110,300))
        py  = cy + int(math.sin(ang) * random.randint(110,300))
        draw.ellipse((px-14,py-14,px+14,py+14), fill=accent)
    draw.rounded_rectangle((330,1180,750,1280), 28, fill=(25,27,34),
                            outline=accent, width=5)


def _draw_tech(draw, size, accent):
    w, h = size
    for x in range(120, w, 150):
        draw.line((x, 360, x, h-360), fill=accent, width=3)
    for y in range(420, h-320, 150):
        draw.line((100, y, w-100, y), fill=accent, width=3)
    for _ in range(24):
        px = random.randint(150, w-150)
        py = random.randint(450, h-450)
        draw.ellipse((px-16,py-16,px+16,py+16), fill=(20,22,27), outline=accent, width=4)


def _draw_history(draw, size, accent):
    w, h = size
    ground = int(h * 0.72)
    draw.rectangle((0, ground, w, h), fill=(24,22,26))
    for x in range(180, w-120, 180):
        top = ground - random.randint(360, 560)
        draw.rectangle((x-42,top,x+42,ground), fill=(38,36,43), outline=accent, width=3)
        draw.rectangle((x-70,top-32,x+70,top), fill=(45,42,50))
    draw.arc((260,ground-600,w-260,ground+120), 180, 360, fill=accent, width=8)


def _draw_default(draw, size, accent):
    w, h = size
    cx, cy = w//2, int(h*0.53)
    draw.ellipse((cx-86,cy-280,cx+86,cy-108), fill=(22,22,28), outline=accent, width=5)
    draw.rounded_rectangle((cx-130,cy-90,cx+130,cy+240), 60, fill=(28,29,37))
    draw.ellipse((cx-300,cy-430,cx+300,cy+170), outline=accent, width=6)


def _add_vignette(img):
    w, h = img.size
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((-220,60,w+220,h-80), fill=225)
    mask = mask.filter(ImageFilter.GaussianBlur(160))
    dark = Image.new("RGB", img.size, (0,0,0))
    return Image.composite(img, dark, mask)


def _pil_art(scene: dict, output_path: str, size=None) -> str:
    """Generate an emotion-themed geometric fallback image using PIL."""
    size    = size or (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
    emotion = scene.get("emotion", "dramatic")
    colors  = _palette(emotion)

    img = _gradient(size, colors).convert("RGBA")
    draw   = ImageDraw.Draw(img, "RGBA")
    accent = colors[2] + (190,)

    text = " ".join([
        scene.get("visual_goal", ""),
        scene.get("narration", ""),
        " ".join(scene.get("queries", [])),
    ]).lower()

    if any(w in text for w in ["phone","text","message","smartphone"]):
        _draw_phone(draw, size[0]//2, int(size[1]*0.48), 1.8, accent)
    elif any(w in text for w in ["science","laboratory","bacteria","microscope","space"]):
        _draw_science(draw, size, accent)
    elif any(w in text for w in ["ai","technology","computer","data","code"]):
        _draw_tech(draw, size, accent)
    elif any(w in text for w in ["history","ancient","ruins","castle","archival"]):
        _draw_history(draw, size, accent)
    elif any(w in text for w in ["argument","couple","wedding","relationship","betrayal"]):
        _draw_people(draw, size, accent)
    else:
        _draw_default(draw, size, accent)

    _add_vignette(img.convert("RGB")).save(output_path, quality=92)
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def generate_fallback_image(scene: dict, output_path: str, size=None) -> str:
    """
    Generate a fallback image (Phase 4 — Grok first).

    Strict tier order when stock assets are rejected by visual verifier:
      1. Grok (xAI)          — best quality "Super Grok" generation
      2. Pollinations (turbo)— fast free fallback
      3. Local PIL art       — always works offline

    Always returns a valid file path — never raises the pipeline.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Tier 1: Grok (xAI) — requires XAI_API_KEY
    result = _generate_with_grok(scene, output_path)
    if result:
        return result

    # Tier 2: Pollinations.ai (turbo model per plan)
    result = _pollinations(scene, output_path)
    if result:
        return result

    # Tier 3: Local PIL geometric art
    print(f"    🖼️  PIL fallback art for: {scene.get('visual_goal','')[:50]}")
    try:
        return _pil_art(scene, output_path, size=size)
    except Exception as e:
        print(f"    ❌ PIL fallback failed ({e}) — creating blank placeholder")
        img = Image.new("RGB", size or (config.VIDEO_WIDTH, config.VIDEO_HEIGHT), (10,10,10))
        img.save(output_path, quality=85)
        return output_path
