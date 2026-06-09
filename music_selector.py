"""
Music Selector — Phase 10: Emotion-Aware Music

Selects background music by deriving the dominant emotion from the
scene plan rather than just the niche mood string.

Emotion → music mood mapping:
  shock / dramatic / tense / anger  →  dramatic
  suspense / mysterious              →  suspenseful
  sadness / emotional / relief       →  emotional
  uplifting / triumphant / calm      →  uplifting
  funny                              →  funny

All tracks are royalty-free / CC0. Cached after first download.
"""

import os
import random
import re

import requests
from bs4 import BeautifulSoup   # Phase 3: Bensound scraper (lightweight)

# ── Royalty-free tracks (CC0 / royalty-free) ─────────────────────────────────
# SoundHelix generates endless generative music under a free license.
# Replace URLs with local files in assets/music/ for offline use.

_TRACKS = {
    "dramatic": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
    ],
    "suspenseful": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
    ],
    "emotional": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
    ],
    "uplifting": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-11.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-14.mp3",
    ],
    "funny": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-13.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-15.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-16.mp3",
    ],
}

# ── Emotion → music mood (Phase 10) ──────────────────────────────────────────
_EMOTION_TO_MOOD = {
    "shock":      "dramatic",
    "dramatic":   "dramatic",
    "tense":      "dramatic",
    "anger":      "dramatic",
    "suspense":   "suspenseful",
    "mysterious": "suspenseful",
    "sadness":    "emotional",
    "emotional":  "emotional",
    "relief":     "emotional",
    "calm":       "uplifting",
    "uplifting":  "uplifting",
    "triumphant": "uplifting",
    "funny":      "funny",
}

# ── Phase 3: Bensound integration ────────────────────────────────────────────
# Static known-good Bensound royalty-free previews (fallback if scraper blocked)
_BENSOUND_STATIC = {
    "dramatic": [
        "https://www.bensound.com/bensound-music/preview/bensound-dramatic.mp3",
        "https://www.bensound.com/bensound-music/preview/bensound-epic.mp3",
    ],
    "suspenseful": [
        "https://www.bensound.com/bensound-music/preview/bensound-suspense.mp3",
        "https://www.bensound.com/bensound-music/preview/bensound-scifi.mp3",
    ],
    "emotional": [
        "https://www.bensound.com/bensound-music/preview/bensound-tenderness.mp3",
        "https://www.bensound.com/bensound-music/preview/bensound-slowmotion.mp3",
    ],
    "uplifting": [
        "https://www.bensound.com/bensound-music/preview/bensound-happyrock.mp3",
        "https://www.bensound.com/bensound-music/preview/bensound-energy.mp3",
    ],
    "funny": [
        "https://www.bensound.com/bensound-music/preview/bensound-funnysong.mp3",
    ],
}

def _scrape_bensound_tracks(mood: str) -> list[str]:
    """
    Lightweight scraper for Bensound royalty-free tracks.
    Tries to find preview mp3 links on category / free music pages.
    Returns list of direct preview URLs (may be empty on anti-bot).
    """
    mood_map = {
        "dramatic": "cinematic",
        "suspenseful": "suspense",
        "emotional": "sad",
        "uplifting": "happy",
        "funny": "funny",
    }
    category = mood_map.get(mood, "cinematic")

    urls_to_try = [
        f"https://www.bensound.com/royalty-free-music/{category}",
        "https://www.bensound.com/free-music",
        f"https://www.bensound.com/royalty-free-music/{category}-music",
    ]

    tracks = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; YTShortsBot/5.0)"}

    for page_url in urls_to_try:
        try:
            resp = requests.get(page_url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Bensound typically has <audio> or buttons with data-preview / src ending in .mp3
            for audio in soup.find_all("audio"):
                src = audio.get("src") or ""
                if src.endswith(".mp3") and "bensound" in src.lower():
                    tracks.append(src)

            # Also look for common preview links
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".mp3") and ("preview" in href.lower() or "bensound" in href.lower()):
                    if not href.startswith("http"):
                        href = "https://www.bensound.com" + href
                    tracks.append(href)

            if tracks:
                break  # found some on this page
        except Exception:
            continue

    # Dedup and limit
    seen = set()
    clean = []
    for t in tracks:
        if t not in seen:
            seen.add(t)
            clean.append(t)
    return clean[:6]


def _get_bensound_for_mood(mood: str) -> str | None:
    """Try dynamic scrape first, fall back to static known tracks."""
    scraped = _scrape_bensound_tracks(mood)
    pool = scraped or _BENSOUND_STATIC.get(mood, _BENSOUND_STATIC["dramatic"])

    if not pool:
        return None

    url = random.choice(pool)
    print(f"    🎵 Bensound candidate: {url}")
    return url


def dominant_emotion(scene_plan: list) -> str:
    """
    Find the most frequent emotion across all scenes.
    Gives 2× weight to the climax beat (highest narrative intensity).
    Returns a music-mood string.
    """
    counts: dict[str, int] = {}
    for scene in scene_plan:
        em     = str(scene.get("emotion", "dramatic")).lower()
        weight = 2 if scene.get("beat") == "climax" else 1
        counts[em] = counts.get(em, 0) + weight

    top_emotion = max(counts, key=counts.get) if counts else "dramatic"
    return _EMOTION_TO_MOOD.get(top_emotion, "dramatic")


def get_music(mood: str, output_path: str = "assets/music.mp3") -> str | None:
    """
    Download (or return cached) music for a given mood string.
    Phase 3: Prefers Bensound (scraped or static) when available,
    otherwise falls back to the original SoundHelix pool.
    """
    os.makedirs("assets", exist_ok=True)

    mood      = str(mood).lower().strip()
    mood      = _EMOTION_TO_MOOD.get(mood, mood)          # normalise if raw emotion
    mood      = mood if mood in _TRACKS else "dramatic"   # safe fallback
    cache     = f"assets/music_{mood}.mp3"

    if os.path.exists(cache):
        print(f"  🎵 Cached {mood} music → {cache}")
        return cache

    # Phase 3: Try Bensound first (scraper + static fallback)
    bensound_url = _get_bensound_for_mood(mood)
    if bensound_url:
        print(f"  🎵 Downloading Bensound {mood} track...")
        try:
            r = requests.get(bensound_url, timeout=30, stream=True,
                             headers={"User-Agent": "YTShortsBot/5.0; +https://github.com/"})
            r.raise_for_status()
            with open(cache, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"     ✅ Bensound saved → {cache}")
            return cache
        except Exception as e:
            print(f"  ⚠️  Bensound download failed ({e}) — falling back to SoundHelix")

    # Original SoundHelix pool
    url = random.choice(_TRACKS[mood])
    print(f"  🎵 Downloading {mood} music (SoundHelix)...")
    try:
        r = requests.get(url, timeout=30, stream=True,
                         headers={"User-Agent": "YTShortsBot/3.0"})
        r.raise_for_status()
        with open(cache, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"     ✅ saved → {cache}")
        return cache
    except Exception as e:
        print(f"  ⚠️  Music download failed: {e}  (video will have voiceover only)")
        return None


def get_music_for_scene_plan(scene_plan: list,
                              niche_mood: str = "dramatic") -> str | None:
    """
    Phase 10: derive music mood from scene plan, fall back to niche_mood.
    Returns a local file path or None.
    """
    music_mood = dominant_emotion(scene_plan) if scene_plan else niche_mood
    print(f"  🎭 Dominant scene emotion → music mood: {music_mood}")
    return get_music(music_mood)
