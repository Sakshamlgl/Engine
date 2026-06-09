import hashlib
import os
import re

import requests
import concurrent.futures

import config
from asset_ranker import rank_candidates
from image_generator import generate_fallback_image
from character_registry import CharacterRegistry
from visual_verifier import visual_verify, log_verification


PIXABAY_IMAGE_URL = "https://pixabay.com/api/"
PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
PEXELS_PHOTO_URL = "https://api.pexels.com/v1/search"
PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
UNSPLASH_PHOTO_URL = "https://api.unsplash.com/search/photos"


def _valid_key(value, placeholder=""):
    return bool(value) and value != placeholder


def _safe(value, limit=48):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_")
    return (value or "asset")[:limit]


def _hash(value):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]


def _request_json(url, params=None, headers=None, timeout=18):
    """Wrapper around requests.get that ensures a descriptive User-Agent.

    Many public APIs (especially Wikimedia) return 403 if no proper
    User-Agent identifying the bot is sent.
    """
    if headers is None:
        headers = {}
    if "User-Agent" not in headers:
        headers["User-Agent"] = "YTShortsBot/5.0 (automated YouTube Shorts generator; https://github.com/)"

    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _best_pixabay_video(hit):
    videos = hit.get("videos", {})
    for quality in ("large", "medium", "small", "tiny"):
        item = videos.get(quality) or {}
        if item.get("url"):
            return item, quality
    return None, None


def _search_pixabay_images(query, per_page=15):
    if not _valid_key(config.PIXABAY_API_KEY, "your_pixabay_api_key"):
        return []

    params = {
        "key": config.PIXABAY_API_KEY,
        "q": query,
        "image_type": "photo",
        "orientation": "all",
        "safesearch": "true",
        "per_page": per_page,
        "min_width": 900,
    }
    data = _request_json(PIXABAY_IMAGE_URL, params=params)
    candidates = []
    for hit in data.get("hits", []):
        url = hit.get("largeImageURL") or hit.get("webformatURL")
        if not url:
            continue
        candidates.append({
            "id": f"pixabay_img_{hit.get('id')}",
            "provider": "pixabay",
            "type": "image",
            "query": query,
            "tags": hit.get("tags", ""),
            "description": hit.get("tags", ""),
            "width": hit.get("imageWidth"),
            "height": hit.get("imageHeight"),
            "download_url": url,
            "page_url": hit.get("pageURL", ""),
            "extension": "jpg",
        })
    return candidates


def _search_pixabay_videos(query, per_page=15):
    if not _valid_key(config.PIXABAY_API_KEY, "your_pixabay_api_key"):
        return []

    params = {
        "key": config.PIXABAY_API_KEY,
        "q": query,
        "video_type": "film",
        "safesearch": "true",
        "per_page": per_page,
        "min_width": 900,
    }
    data = _request_json(PIXABAY_VIDEO_URL, params=params)
    candidates = []
    for hit in data.get("hits", []):
        video, quality = _best_pixabay_video(hit)
        if not video:
            continue
        candidates.append({
            "id": f"pixabay_vid_{hit.get('id')}_{quality}",
            "provider": "pixabay",
            "type": "video",
            "query": query,
            "tags": hit.get("tags", ""),
            "description": hit.get("tags", ""),
            "width": video.get("width"),
            "height": video.get("height"),
            "duration": hit.get("duration"),
            "download_url": video.get("url"),
            "page_url": hit.get("pageURL", ""),
            "extension": "mp4",
        })
    return candidates


def _search_pexels_photos(query, per_page=15):
    if not _valid_key(config.PEXELS_API_KEY):
        return []

    params = {"query": query, "per_page": per_page, "orientation": "portrait"}
    headers = {"Authorization": config.PEXELS_API_KEY}
    data = _request_json(PEXELS_PHOTO_URL, params=params, headers=headers)
    candidates = []
    for photo in data.get("photos", []):
        src = photo.get("src", {})
        url = src.get("large2x") or src.get("large") or src.get("original")
        if not url:
            continue
        candidates.append({
            "id": f"pexels_img_{photo.get('id')}",
            "provider": "pexels",
            "type": "image",
            "query": query,
            "tags": photo.get("alt", ""),
            "description": photo.get("alt", ""),
            "width": photo.get("width"),
            "height": photo.get("height"),
            "download_url": url,
            "page_url": photo.get("url", ""),
            "extension": "jpg",
        })
    return candidates


def _best_pexels_video_file(video):
    files = video.get("video_files", [])
    if not files:
        return None
    files = sorted(files, key=lambda item: (item.get("width") or 0) * (item.get("height") or 0), reverse=True)
    for item in files:
        if item.get("link") and "video" in str(item.get("file_type", "")):
            return item
    return files[0]


def _search_pexels_videos(query, per_page=15):
    if not _valid_key(config.PEXELS_API_KEY):
        return []

    params = {"query": query, "per_page": per_page, "orientation": "portrait"}
    headers = {"Authorization": config.PEXELS_API_KEY}
    data = _request_json(PEXELS_VIDEO_URL, params=params, headers=headers)
    candidates = []
    for video in data.get("videos", []):
        video_file = _best_pexels_video_file(video)
        if not video_file or not video_file.get("link"):
            continue
        candidates.append({
            "id": f"pexels_vid_{video.get('id')}",
            "provider": "pexels",
            "type": "video",
            "query": query,
            "tags": query,
            "description": query,
            "width": video_file.get("width") or video.get("width"),
            "height": video_file.get("height") or video.get("height"),
            "duration": video.get("duration"),
            "download_url": video_file.get("link"),
            "page_url": video.get("url", ""),
            "extension": "mp4",
        })
    return candidates


def _search_unsplash_photos(query, per_page=15):
    if not _valid_key(config.UNSPLASH_ACCESS_KEY):
        return []

    params = {
        "query": query,
        "per_page": per_page,
        "orientation": "portrait",
        "client_id": config.UNSPLASH_ACCESS_KEY,
    }
    data = _request_json(UNSPLASH_PHOTO_URL, params=params)
    candidates = []
    for photo in data.get("results", []):
        urls = photo.get("urls", {})
        url = urls.get("regular") or urls.get("full")
        if not url:
            continue
        description = photo.get("alt_description") or photo.get("description") or query
        candidates.append({
            "id": f"unsplash_img_{photo.get('id')}",
            "provider": "unsplash",
            "type": "image",
            "query": query,
            "tags": description,
            "description": description,
            "width": photo.get("width"),
            "height": photo.get("height"),
            "download_url": url,
            "page_url": (photo.get("links") or {}).get("html", ""),
            "extension": "jpg",
        })
    return candidates


# ── NEW Phase 2: NASA Images API (public, no key) ─────────────────────────────
def _search_nasa_images(query, per_page=15):
    """Search NASA's public images API for relevant photos."""
    url = "https://images-api.nasa.gov/search"
    params = {"q": query, "media_type": "image"}
    try:
        data = _request_json(url, params=params)
        items = data.get("collection", {}).get("items", [])[:per_page]
        candidates = []
        for item in items:
            links = item.get("links", [])
            if not links:
                continue
            href = links[0].get("href")
            if not href:
                continue
            data_info = (item.get("data") or [{}])[0]
            title = data_info.get("title", query)[:80]
            desc = (data_info.get("description") or "")[:180]
            candidates.append({
                "id": f"nasa_{data_info.get('nasa_id', _hash(href))}",
                "provider": "nasa",
                "type": "image",
                "query": query,
                "tags": title,
                "description": desc or title,
                "download_url": href,
                "page_url": "https://images.nasa.gov",
                "extension": "jpg",
            })
        return candidates
    except Exception as exc:
        print(f"    NASA search failed for '{query}' ({exc})")
        return []


# ── NEW Phase 2: Wikimedia Commons (Wikipedia) images (public) ────────────────
def _search_wikipedia_images(query, per_page=15):
    """Search Wikimedia Commons for free images via their API.

    Wikimedia requires a descriptive User-Agent (per their policy).
    Without it, requests are rejected with 403 Forbidden.
    """
    api_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": per_page,
        "prop": "imageinfo",
        "iiprop": "url|size",
        "iiurlwidth": 900,
        "gsrnamespace": 6,  # File namespace only
    }
    # Critical: Wikimedia blocks requests without a proper identifying User-Agent
    headers = {
        "User-Agent": "YTShortsBot/5.0 (automated YouTube Shorts generator; https://github.com/)"
    }
    try:
        data = _request_json(api_url, params=params, headers=headers)
        pages = data.get("query", {}).get("pages", {})
        candidates = []
        for page_id, page in pages.items():
            if "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            title = page.get("title", "").replace("File:", "")
            candidates.append({
                "id": f"wikimedia_{page.get('pageid')}",
                "provider": "wikimedia",
                "type": "image",
                "query": query,
                "tags": title,
                "description": title,
                "width": info.get("width"),
                "height": info.get("height"),
                "download_url": url,
                "page_url": f"https://commons.wikimedia.org/wiki/{page.get('title', '')}",
                "extension": "jpg",
            })
        return candidates
    except Exception as exc:
        # 403 usually means missing/invalid User-Agent (Wikimedia policy).
        # We now always send one via _request_json.
        print(f"    Wikimedia search failed for '{query}' ({exc})")
        return []


def _run_search(searcher, query):
    try:
        return searcher(query)
    except Exception as exc:
        print(f"    {searcher.__name__} failed for '{query}' ({exc})")
        return []

def _search_candidates_parallel(queries, media_types):
    tasks = []
    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for query in queries:
            for media_type in media_types:
                searchers = []
                if media_type == "image":
                    # Phase 2: Added NASA and Wikimedia Commons public sources
                    searchers = [
                        _search_pixabay_images,
                        _search_pexels_photos,
                        _search_unsplash_photos,
                        _search_nasa_images,
                        _search_wikipedia_images,
                    ]
                elif media_type == "video":
                    searchers = [_search_pixabay_videos, _search_pexels_videos]
                for searcher in searchers:
                    tasks.append(executor.submit(_run_search, searcher, query))
        
        for future in concurrent.futures.as_completed(tasks):
            candidates.extend(future.result())
            
    return candidates


def _download_candidate(candidate, scene_index):
    os.makedirs(config.SCENE_ASSET_DIR, exist_ok=True)
    ext = candidate.get("extension") or ("mp4" if candidate.get("type") == "video" else "jpg")
    filename = f"scene_{scene_index:02d}_{candidate['provider']}_{_safe(candidate['id'])}_{_hash(candidate.get('download_url'))}.{ext}"
    path = os.path.join(config.SCENE_ASSET_DIR, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(candidate["download_url"], stream=True, timeout=75, headers=headers) as response:
        response.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)

    if os.path.getsize(path) <= 0:
        raise RuntimeError(f"Downloaded empty asset: {path}")
    return path


def _fallback_asset(scene, scene_index):
    os.makedirs(config.SCENE_ASSET_DIR, exist_ok=True)
    filename = f"scene_{scene_index:02d}_generated_{_safe(scene.get('visual_goal', 'fallback'))}_{_hash(scene)}.jpg"
    path = os.path.join(config.SCENE_ASSET_DIR, filename)
    if not os.path.exists(path):
        generate_fallback_image(scene, path)
    return {
        **scene,
        "asset": path,
        "type": "image",
        "provider": "generated",
        "source_url": "",
        "query": (scene.get("queries") or [""])[0],
        "score": 0,
        "fallback": True,
    }


def _scene_asset_payload(scene, candidate, local_path):
    return {
        **scene,
        "asset": local_path,
        "type": candidate.get("type"),
        "provider": candidate.get("provider"),
        "source_url": candidate.get("page_url", ""),
        "query": candidate.get("query", ""),
        "score": candidate.get("score", 0),
        "fallback": False,
        "tags": candidate.get("tags", ""),
        "description": candidate.get("description", ""),
    }


def fetch_scene_asset(scene, scene_index, previous_assets=None, character_registry=None):
    queries = scene.get("queries") or [scene.get("visual_goal", "dramatic story")]
    media_types = scene.get("asset_types") or ["image", "video"]
    candidates = []

    print(f"  Scene {scene_index}: searching assets for {scene.get('beat', 'scene')}...")
    candidates = _search_candidates_parallel(queries, media_types)

    if candidates:
        ranked = rank_candidates(scene, candidates, previous_assets=previous_assets, character_registry=character_registry)
        for candidate in ranked:
            if candidate.get("score", 0) < config.ASSET_SCORE_THRESHOLD:
                continue
            try:
                path = _download_candidate(candidate, scene_index)

                # Visual verification — check the asset actually matches the scene
                passes, v_score, v_reason = visual_verify(scene, path)
                log_verification(scene_index, candidate, passes, v_score, v_reason)

                # Phase 4: Strict vision threshold enforcement.
                # If score is too low we reject even if visual_verify returned passes
                # (due to its internal threshold) and fall through to AI image generator.
                threshold = getattr(config, "VISUAL_VERIFY_THRESHOLD", 50)
                effective_pass = passes and (v_score is None or v_score > threshold)

                if not effective_pass:
                    print(f"    ⚠️  vision score {v_score} <= {threshold} — rejecting for AI fallback")
                    continue

                print(
                    f"    selected {candidate['provider']} {candidate['type']} "
                    f"score={candidate.get('score')} vision={v_score} query='{candidate.get('query')}'"
                )
                # Phase 12: record this asset for character consistency
                if character_registry is not None:
                    character_registry.record(scene.get("character", ""), candidate)
                payload = _scene_asset_payload(scene, candidate, path)
                payload["vision_score"] = v_score
                return payload
            except Exception as exc:
                print(f"    download failed for {candidate.get('provider')} ({exc})")

    print("    no suitable stock asset found; generating local fallback image")
    return _fallback_asset(scene, scene_index)


def fetch_assets(scene_plan, character_registry=None):
    """Fetch or generate one visual asset per scene.

    character_registry: optional CharacterRegistry for Phase 12 consistency.
    If None, a fresh one is created automatically.
    """
    if character_registry is None:
        character_registry = CharacterRegistry()   # Phase 12: one registry per video

    assets = []
    for index, scene in enumerate(scene_plan, start=1):
        asset = fetch_scene_asset(
            scene, index,
            previous_assets=assets,
            character_registry=character_registry,  # Phase 12
        )
        assets.append(asset)

    # Log character usage summary
    summary = character_registry.summary()
    if summary:
        print("  🎭 Character registry:")
        for char, info in summary.items():
            print(f"     {char}: {info['appearances']} scene(s) → {info['queries_used']}")

    return assets