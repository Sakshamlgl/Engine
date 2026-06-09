import os
import requests
import random
import config

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "your_pixabay_api_key")
VIDEO_API_URL   = "https://pixabay.com/api/videos/"

# ── Per-niche search keywords ──────────────────────────────
NICHE_KEYWORDS = {
    "A": ["minecraft parkour", "subway surfers", "satisfying gameplay",
          "temple run", "geometry dash"],
    "B": ["space galaxy stars", "universe cosmos", "planet earth",
          "microscope science", "nature timelapse", "ocean deep sea"],
    "C": ["technology digital", "computer code", "futuristic city",
          "data network", "circuit board", "artificial intelligence"],
    "D": ["ancient ruins", "medieval castle", "historical city",
          "old film vintage", "roman architecture", "egyptian pyramids"],
}


def fetch_pixabay_video(niche_key, min_duration=30):
    """
    Search Pixabay for a royalty-free background video matching the niche.
    Downloads to assets/ and returns the local path.
    Caches — won't re-download if already saved.
    """
    niche_key = niche_key.upper()
    keywords  = NICHE_KEYWORDS.get(niche_key, NICHE_KEYWORDS["A"])
    keyword   = random.choice(keywords)
    safe_name = keyword.replace(" ", "_")
    out_path  = f"assets/bg_{niche_key}_{safe_name}.mp4"

    # Return cached file if exists
    if os.path.exists(out_path):
        print(f"  Using cached background: {out_path}")
        return out_path

    print(f"  Searching Pixabay for '{keyword}'...")
    params = {
        "key":        PIXABAY_API_KEY,
        "q":          keyword,
        "video_type": "film",
        "order":      "popular",
        "safesearch": "true",
        "per_page":   20,
        "min_width":  1280,
    }

    try:
        r = requests.get(VIDEO_API_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", [])

        if not hits:
            print(f"  No Pixabay results for '{keyword}'")
            return None

        # Filter for videos at least min_duration seconds
        valid = [h for h in hits if h.get("duration", 0) >= min_duration]
        if not valid:
            valid = hits  # fallback: use any

        video = random.choice(valid[:10])

        # Pick best quality available (prefer large > medium > small)
        videos   = video.get("videos", {})
        url      = None
        quality  = None

        for q in ["large", "medium", "small", "tiny"]:
            v = videos.get(q, {})
            if v.get("url") and v.get("size", 0) > 0:
                url     = v["url"]
                quality = q
                break

        if not url:
            print("  No downloadable URL found")
            return None

        print(f"  Downloading '{keyword}' ({quality}) from Pixabay...")
        dl = requests.get(url, stream=True, timeout=60,
                          headers={"User-Agent": "Mozilla/5.0"})
        dl.raise_for_status()

        os.makedirs("assets", exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=65536):
                f.write(chunk)

        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  ✅ Downloaded → {out_path} ({size_mb:.1f} MB)")
        return out_path

    except Exception as e:
        print(f"  ❌ Pixabay fetch failed: {e}")
        return None


def get_background_video(niche_key):
    """
    Get background video for a niche.
    Priority: 1) Pixabay download  2) existing local file  3) default gameplay
    """
    from niche_config import get_niche
    niche    = get_niche(niche_key)
    bg_video = niche["bg_video"]

    # Try Pixabay first
    if PIXABAY_API_KEY != "your_pixabay_api_key":
        pixabay_path = fetch_pixabay_video(niche_key)
        if pixabay_path:
            return pixabay_path

    # Fall back to locally saved niche video
    if os.path.exists(bg_video):
        print(f"  Using local background: {bg_video}")
        return bg_video

    # Final fallback: default gameplay
    if os.path.exists(config.GAMEPLAY_VIDEO_PATH):
        print(f"  ⚠️  Falling back to gameplay.mp4")
        return config.GAMEPLAY_VIDEO_PATH

    return None
