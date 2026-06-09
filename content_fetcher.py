import requests
import xml.etree.ElementTree as ET
import random
import re
import config

HEADERS = {"User-Agent": config.REDDIT_USER_AGENT}


# ── Helpers ────────────────────────────────────────────────

def strip_html(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def clean(text):
    return re.sub(r"\s+", " ", strip_html(text)).strip()


# ── NICHE A: Reddit RSS ────────────────────────────────────

def fetch_reddit(used_ids=set()):
    subreddit = random.choice(config.SUBREDDITS)
    url       = f"https://www.reddit.com/r/{subreddit}/top.rss?t=day&limit=10"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ❌ Reddit RSS failed: {e}")
        return None

    root    = ET.fromstring(r.content)
    ns      = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    posts   = []

    for entry in entries:
        title   = entry.findtext("atom:title", default="", namespaces=ns).strip()
        content = clean(entry.findtext("atom:content", default="", namespaces=ns))
        link    = entry.findtext("atom:link", default="", namespaces=ns).strip()
        pid     = entry.findtext("atom:id", default="", namespaces=ns).strip().split("_")[-1]
        content = content[:config.MAX_POST_LENGTH]
        if not content or len(content) < 100 or pid in used_ids:
            continue
        posts.append({"id": pid, "title": title, "content": content,
                      "subreddit": subreddit, "url": link, "source": "reddit"})

    if not posts:
        print(f"  No fresh Reddit posts from r/{subreddit}, trying weekly...")
        return fetch_reddit_weekly(subreddit, used_ids)

    print(f"  Found {len(posts)} Reddit posts from r/{subreddit}")
    return posts[0] if posts else None


def fetch_reddit_weekly(subreddit, used_ids=set()):
    url = f"https://www.reddit.com/r/{subreddit}/top.rss?t=week&limit=10"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        root    = ET.fromstring(r.content)
        ns      = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        for entry in entries:
            title   = entry.findtext("atom:title", default="", namespaces=ns).strip()
            content = clean(entry.findtext("atom:content", default="", namespaces=ns))
            link    = entry.findtext("atom:link", default="", namespaces=ns).strip()
            pid     = entry.findtext("atom:id", default="", namespaces=ns).strip().split("_")[-1]
            content = content[:config.MAX_POST_LENGTH]
            if content and len(content) >= 100 and pid not in used_ids:
                return {"id": pid, "title": title, "content": content,
                        "subreddit": subreddit, "url": link, "source": "reddit"}
    except Exception as e:
        print(f"  ❌ Reddit weekly RSS failed: {e}")
    return None


# ── NICHE B: Science (arXiv + Wikipedia) ──────────────────

def fetch_science(used_ids=set()):
    # Try arXiv first (cs.AI, physics, astro-ph, q-bio)
    categories = ["cs.AI", "astro-ph", "physics", "q-bio"]
    cat        = random.choice(categories)
    url        = f"https://export.arxiv.org/rss/{cat}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        root    = ET.fromstring(r.content)
        items   = root.findall(".//item")
        random.shuffle(items)
        for item in items[:10]:
            title   = clean(item.findtext("title", default=""))
            desc    = clean(item.findtext("description", default=""))
            link    = item.findtext("link", default="")
            pid     = re.sub(r"[^a-z0-9]", "", link.lower())[-20:]
            if pid in used_ids or len(desc) < 80:
                continue
            print(f"  Found arXiv paper: {title[:60]}...")
            return {"id": pid, "title": title, "content": desc[:1500],
                    "subreddit": f"arXiv/{cat}", "url": link, "source": "arxiv"}
    except Exception as e:
        print(f"  arXiv failed ({e}), trying Wikipedia science...")

    # Fallback: Wikipedia featured science article
    return fetch_wikipedia("science")


# ── NICHE C: AI News RSS ───────────────────────────────────

AI_RSS_FEEDS = [
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://venturebeat.com/ai/feed/",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "https://feeds.feedburner.com/TheHackersNews",
]

def fetch_ai_news(used_ids=set()):
    feeds = AI_RSS_FEEDS.copy()
    random.shuffle(feeds)
    for feed_url in feeds:
        try:
            r = requests.get(feed_url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            root  = ET.fromstring(r.content)
            items = root.findall(".//item")
            random.shuffle(items)
            for item in items[:15]:
                title   = clean(item.findtext("title", default=""))
                desc    = clean(item.findtext("description", default=""))
                link    = item.findtext("link", default="")
                pid     = re.sub(r"[^a-z0-9]", "", link.lower())[-20:]
                # Only AI-related
                if not any(kw in (title+desc).lower() for kw in
                           ["ai", "artificial intelligence", "gpt", "llm",
                            "openai", "anthropic", "gemini", "model", "robot"]):
                    continue
                if pid in used_ids or len(desc) < 80:
                    continue
                print(f"  Found AI news: {title[:60]}...")
                return {"id": pid, "title": title, "content": desc[:1500],
                        "subreddit": "AI News", "url": link, "source": "ai_rss"}
        except Exception as e:
            print(f"  Feed {feed_url} failed: {e}")
            continue
    return None


# ── NICHE D: History (Wikipedia On This Day) ──────────────

def fetch_history(used_ids=set()):
    return fetch_wikipedia("history", used_ids)

def fetch_wikipedia(topic="history", used_ids=None):
    if used_ids is None:
        used_ids = set()
    """Fetch a random featured Wikipedia article for the given topic."""
    search_terms = {
        "history":  ["historical mystery", "ancient civilization", "lost empire",
                     "unsolved history", "forgotten history"],
        "science":  ["scientific discovery", "quantum physics",
                     "space exploration", "biology breakthrough"],
    }
    term = random.choice(search_terms.get(topic, ["interesting facts"]))

    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": term,
        "gsrlimit": "10", "prop": "extracts",
        "exintro": True, "explaintext": True,
        "exsentences": "8"
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=12)
        r.raise_for_status()
        data  = r.json()
        pages = list(data.get("query", {}).get("pages", {}).values())
        random.shuffle(pages)
        for page in pages:
            extract = page.get("extract", "").strip()
            title   = page.get("title", "")
            pid     = str(page.get("pageid", ""))
            if pid in used_ids or len(extract) < 100:
                continue
            wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            print(f"  Found Wikipedia: {title}")
            return {"id": pid, "title": title, "content": extract[:1500],
                    "subreddit": f"Wikipedia/{topic}", "url": wiki_url, "source": "wikipedia"}
    except Exception as e:
        print(f"  Wikipedia failed: {e}")
    return None


# ── Main dispatcher ────────────────────────────────────────

def fetch_content(niche_key, used_ids=set()):
    """Fetch content for the given niche. Returns a post dict or None."""
    key = niche_key.upper()
    if key == "A":
        return fetch_reddit(used_ids)
    elif key == "B":
        return fetch_science(used_ids)
    elif key == "C":
        return fetch_ai_news(used_ids)
    elif key == "D":
        return fetch_history(used_ids)
    else:
        raise ValueError(f"Unknown niche: {niche_key}")
