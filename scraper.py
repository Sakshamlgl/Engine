import requests
import xml.etree.ElementTree as ET
import config


def get_top_posts(subreddit_name, limit=10):
    """Scrape top posts from a subreddit using Reddit's public RSS feed. No API key needed."""

    url = f"https://www.reddit.com/r/{subreddit_name}/top.rss?t=day&limit={limit}"
    headers = {"User-Agent": config.REDDIT_USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ Failed to fetch RSS for r/{subreddit_name}: {e}")
        return []

    # Parse RSS/Atom feed
    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    posts = []
    for entry in entries:
        title   = entry.findtext("atom:title", default="", namespaces=ns).strip()
        content = entry.findtext("atom:content", default="", namespaces=ns).strip()
        link    = entry.findtext("atom:link", default="", namespaces=ns).strip()
        post_id = entry.findtext("atom:id", default="", namespaces=ns).strip().split("_")[-1]

        # Strip HTML tags from content
        import re
        content = re.sub(r"<[^>]+>", "", content).strip()
        content = content[:config.MAX_POST_LENGTH]

        # Skip posts with no real text body
        if not content or len(content) < 100:
            continue

        posts.append({
            "id":        post_id,
            "title":     title,
            "content":   content,
            "score":     0,       # RSS doesn't expose score
            "subreddit": subreddit_name,
            "url":       link
        })

    # Fallback to weekly if nothing today
    if not posts:
        print(f"  No valid posts today in r/{subreddit_name}, trying this week...")
        return get_top_posts_weekly(subreddit_name, limit)

    print(f"  Found {len(posts)} valid posts from r/{subreddit_name} (via RSS)")
    return posts


def get_top_posts_weekly(subreddit_name, limit=10):
    """Weekly fallback using RSS."""
    url = f"https://www.reddit.com/r/{subreddit_name}/top.rss?t=week&limit={limit}"
    headers = {"User-Agent": config.REDDIT_USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ❌ Failed to fetch weekly RSS for r/{subreddit_name}: {e}")
        return []

    import re
    root    = ET.fromstring(response.content)
    ns      = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    posts   = []

    for entry in entries:
        title   = entry.findtext("atom:title", default="", namespaces=ns).strip()
        content = entry.findtext("atom:content", default="", namespaces=ns).strip()
        link    = entry.findtext("atom:link", default="", namespaces=ns).strip()
        post_id = entry.findtext("atom:id", default="", namespaces=ns).strip().split("_")[-1]

        content = re.sub(r"<[^>]+>", "", content).strip()
        content = content[:config.MAX_POST_LENGTH]

        if not content or len(content) < 100:
            continue

        posts.append({
            "id":        post_id,
            "title":     title,
            "content":   content,
            "score":     0,
            "subreddit": subreddit_name,
            "url":       link
        })

    print(f"  Found {len(posts)} valid posts from r/{subreddit_name} (weekly RSS fallback)")
    return posts
