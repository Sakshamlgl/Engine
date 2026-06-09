import os
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════
# API KEYS  — set in .env, never commit .env to git
# ══════════════════════════════════════════════════════════

REDDIT_USER_AGENT    = "YTShortsBot/4.0"
GROQ_API_KEY         = os.getenv("GROQ_API_KEY",          "your_groq_api_key")
PIXABAY_API_KEY      = os.getenv("PIXABAY_API_KEY",        "")
PEXELS_API_KEY       = os.getenv("PEXELS_API_KEY",         "")
UNSPLASH_ACCESS_KEY  = os.getenv("UNSPLASH_ACCESS_KEY",    "")
HF_TOKEN             = os.getenv("HF_TOKEN",               "")
XAI_API_KEY          = os.getenv("XAI_API_KEY",            "")  # for Grok image generation (Phase 4)

# ══════════════════════════════════════════════════════════
# FEATURE FLAGS — Phase 4 base
# ══════════════════════════════════════════════════════════

# Phase 4 — LLM re-ranking (costs extra Groq tokens, improves relevance)
LLM_ASSET_RANKING    = os.getenv("LLM_ASSET_RANKING",    "false").lower() == "true"

# Phase 14 — Pollinations.ai free AI image generation fallback
POLLINATIONS_ENABLED = os.getenv("POLLINATIONS_ENABLED", "true").lower()  != "false"
POLLINATIONS_TIMEOUT = int(os.getenv("POLLINATIONS_TIMEOUT", "25"))        # seconds

# Phase 15 — Image-to-Video (currently placeholder; set provider when available)
# Supported providers (future / paid): "kling" | "runway" | "luma" | None
IMAGE_TO_VIDEO_PROVIDER = os.getenv("IMAGE_TO_VIDEO_PROVIDER", None)

# ══════════════════════════════════════════════════════════
# NEW v4 IMPROVEMENT FLAGS
# ══════════════════════════════════════════════════════════

# Story Brain — LLM feedback loop that scores every fetched asset and
# re-fetches weak ones with smarter queries. Needs GROQ_API_KEY.
STORY_BRAIN_ENABLED      = os.getenv("STORY_BRAIN_ENABLED",      "true").lower() != "false"
STORY_BRAIN_THRESHOLD    = int(os.getenv("STORY_BRAIN_THRESHOLD",    "70"))   # 0-100
STORY_BRAIN_MAX_RETRIES  = int(os.getenv("STORY_BRAIN_MAX_RETRIES",  "2"))    # retry attempts

# Shot Planning — expands each scene into professional multi-shot sequences
# (wide + close-up + reaction + detail) instead of 1 scene = 1 asset.
SHOT_PLANNING_ENABLED    = os.getenv("SHOT_PLANNING_ENABLED",    "true").lower() != "false"
SHOT_PLANNING_MAX        = int(os.getenv("SHOT_PLANNING_MAX",        "4"))    # max shots per scene

# Sequencer — enforces image→video→image alternating rhythm and climax
# sandwiches (image → climax video → image). Applied before AND after fetch.
SEQUENCER_ENABLED         = os.getenv("SEQUENCER_ENABLED",         "true").lower() != "false"
SEQUENCER_VIDEO_CADENCE   = int(os.getenv("SEQUENCER_VIDEO_CADENCE",   "2"))  # every Nth scene = video
SEQUENCER_CLIMAX_SANDWICH = os.getenv("SEQUENCER_CLIMAX_SANDWICH", "true").lower() != "false"

# Visual Verifier — uses Groq's free vision model to check each downloaded
# asset actually matches the scene narration (rejects bad matches).
VISUAL_VERIFY_ENABLED     = os.getenv("VISUAL_VERIFY_ENABLED",     "true").lower() != "false"
VISUAL_VERIFY_THRESHOLD   = int(os.getenv("VISUAL_VERIFY_THRESHOLD",   "50"))  # 0-100 (raised in Phase 4 for stricter quality + AI fallback)

# ══════════════════════════════════════════════════════════
# CONTENT SOURCES
# ══════════════════════════════════════════════════════════

SUBREDDITS = [
    # Drama / AITA
    "AmItheAsshole", "AITAH", "relationship_advice",
    "confessions",   "TrueOffMyChest", "offmychest",
    "tifu",          "entitledparents", "pettyrevenge",
    "ProRevenge",    "raisedbynarcissists", "ChoosingBeggars",
    "weddingshaming",
    # Horror / paranormal
    "nosleep",       "Thetruthishere",  "Ghoststories",
    "ParanormalEncounters", "letsnotmeet", "creepyencounters",
    "glitch_in_the_matrix", "unexplained",
]

MAX_POST_LENGTH = 3000

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════

OUTPUT_DIR          = "output"
USED_IDS_PATH       = "output/used_ids.txt"
SCENE_ASSET_DIR     = "output/scene_assets"
GAMEPLAY_VIDEO_PATH = "assets/gameplay.mp4"   # legacy fallback only

# ══════════════════════════════════════════════════════════
# VIDEO FORMAT (YouTube Shorts — 9:16)
# ══════════════════════════════════════════════════════════

VIDEO_WIDTH    = 1080
VIDEO_HEIGHT   = 1920
TARGET_SECONDS = 45

# ══════════════════════════════════════════════════════════
# SCENE PLANNING
# ══════════════════════════════════════════════════════════

SCENE_PLAN_MIN = 4    # minimum scenes per video
SCENE_PLAN_MAX = 12   # maximum scenes per video

# ══════════════════════════════════════════════════════════
# ASSET FETCHING & RANKING
# ══════════════════════════════════════════════════════════

ASSET_CANDIDATE_LIMIT = 30   # max candidates collected before ranking
ASSET_SCORE_THRESHOLD = 18   # minimum heuristic score to attempt download

# Phase 7 — 70 % images, 30 % videos (sequencer may override per-scene)
PREFERRED_IMAGE_RATIO = 0.70

# ══════════════════════════════════════════════════════════
# v5 UPGRADE FLAGS
# ══════════════════════════════════════════════════════════

# ── CLIP Retrieval (highest ROI upgrade) ───────────────────
# Adds semantic embedding similarity on top of keyword ranking.
# Requires: pip install sentence-transformers  (or open_clip_torch)
# Degrades gracefully to heuristic-only if not installed.
CLIP_ENABLED = os.getenv("CLIP_ENABLED", "true").lower() != "false"
CLIP_WEIGHT  = float(os.getenv("CLIP_WEIGHT", "0.40"))   # CLIP weight vs heuristic

# ── Story Brain Vision (major quality jump) ────────────────
# Story Brain now SEES the actual image when scoring assets.
# Uses Groq llama-3.2-11b-vision (same key, no extra cost).
STORY_BRAIN_VISION_ENABLED = os.getenv("STORY_BRAIN_VISION_ENABLED", "true").lower() != "false"

# ── Editing Intelligence ───────────────────────────────────
# Annotates shots with professional cut decisions:
# pacing, ken burns direction, hold frames, reaction_first, etc.
EDITING_INTELLIGENCE_ENABLED = os.getenv("EDITING_INTELLIGENCE_ENABLED", "true").lower() != "false"
# Use LLM for nuanced decisions on climax/discovery beats only
EDITING_INTELLIGENCE_LLM = os.getenv("EDITING_INTELLIGENCE_LLM", "true").lower() != "false"

# ── Character Face Embeddings ──────────────────────────────
# Character consistency via embeddings instead of keyword overlap.
# Auto-selects best available: InsightFace > CLIP > MiniLM > keyword
# InsightFace: pip install insightface  (optional, highest accuracy)
CHARACTER_EMBED_ENABLED = os.getenv("CHARACTER_EMBED_ENABLED", "true").lower() != "false"
