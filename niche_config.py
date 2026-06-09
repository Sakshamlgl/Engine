# ============================================================
# NICHE CONFIGURATION
# Each niche has its own: sources, tone, background, music,
# hashtags, title style, and description style
# ============================================================

NICHES = {
    "A": {
        "name":        "Reddit Stories",
        "emoji":       "🔥",
        "tone":        "dramatic",
        "bg_video":    "assets/gameplay.mp4",       # Minecraft/Subway Surfers
        "music_mood":  None,                        # auto from script mood
        "sources":     ["reddit"],
        "hashtags_base": [
            "#RedditStories", "#AITA", "#Shorts",
            "#RedditTea", "#AITAH", "#Storytime",
            "#RelationshipAdvice", "#RedditDrama"
        ],
        "title_style": "curiosity_gap",             # "she did X and it destroyed everything"
        "desc_style":  "story_tease",               # tease + CTA + hashtags
    },
    "B": {
        "name":        "Science Facts",
        "emoji":       "🔬",
        "tone":        "mind_blowing",
        "bg_video":    "assets/space.mp4",          # space/nature footage
        "music_mood":  "uplifting",
        "sources":     ["arxiv", "wikipedia_science"],
        "hashtags_base": [
            "#ScienceFacts", "#DidYouKnow", "#Shorts",
            "#Science", "#Physics", "#SpaceFacts",
            "#MindBlown", "#LearnOnTikTok"
        ],
        "title_style": "mind_blow",                 # "Scientists just discovered X and it changes everything"
        "desc_style":  "fact_expand",               # expand the fact + source + CTA
    },
    "C": {
        "name":        "AI News",
        "emoji":       "🤖",
        "tone":        "urgent",
        "bg_video":    "assets/tech.mp4",           # code/tech visuals
        "music_mood":  "dramatic",
        "sources":     ["ai_rss"],
        "hashtags_base": [
            "#AINews", "#ArtificialIntelligence", "#Shorts",
            "#Tech", "#ChatGPT", "#MachineLearning",
            "#FutureOfAI", "#TechNews"
        ],
        "title_style": "urgency",                   # "This AI just did X — the world is not ready"
        "desc_style":  "news_breakdown",            # what happened + why it matters + CTA
    },
    "D": {
        "name":        "History Mysteries",
        "emoji":       "🏛️",
        "tone":        "mysterious",
        "bg_video":    "assets/oldfilm.mp4",        # vintage/documentary footage
        "music_mood":  "suspenseful",
        "sources":     ["wikipedia_history", "ai_history"],
        "hashtags_base": [
            "#HistoryMysteriesShorts", "#History", "#Shorts",
            "#HistoryFacts", "#Mystery", "#AncientHistory",
            "#HiddenHistory", "#HistoryShorts"
        ],
        "title_style": "mystery",                   # "The [X] that historians still can't explain"
        "desc_style":  "mystery_tease",             # hook + historical context + open question + CTA
    },
}


def get_niche(niche_key):
    key = niche_key.upper()
    if key not in NICHES:
        raise ValueError(f"Unknown niche '{niche_key}'. Choose from: {list(NICHES.keys())}")
    return NICHES[key]
