# YouTube Shorts Bot v4 — Story Brain Edition

Automated story-to-video pipeline that turns Reddit posts into publish-ready YouTube Shorts.

## What's new in v4

v3 had five structural weaknesses. v4 fixes all of them:

| # | Problem | v4 Fix |
|---|---------|--------|
| 1 | No central story brain — no feedback loop | **Story Brain** — LLM scores every fetched asset, re-fetches weak ones with smarter queries |
| 2 | Search queries from keyword rules (`if "phone": return "person reading phone"`) | **AI queries** — LLM deeply reads the scene's role in the story arc and generates concrete, cinematically-grounded searches |
| 3 | No visual verification — assumed search result = correct | **Visual Verifier** — Groq vision model physically looks at each downloaded asset and rejects mismatches |
| 4 | No shot planning — 1 scene = 1 asset | **Shot Planner** — each scene becomes wide + close-up + reaction + detail shots like real short-form video |
| 5 | Flat image-image-video-image-image-video sequence | **Sequencer** — deliberate image→video→image rhythm with climax sandwiches (image→climax video→image) |

## Architecture

```
Phase 1   Content fetching
Phase 2   Script generation
Phase 3   Scene planning          (LLM director + AI story-aware queries)       ← upgraded
Phase 3b  Shot planning ★         (wide + close + reaction + detail)             ← NEW
Phase 3c  Pre-fetch sequencing ★  (image→video rhythm plan applied before fetch) ← NEW
Phase 4   Asset fetching          (multi-query · ranking · vision verify ★)      ← upgraded
Phase 4b  Story Brain review ★    (LLM scores every asset, re-fetches weak ones) ← NEW
Phase 4c  Post-fetch sequencing ★ (enforce rhythm after fetch, fix fallbacks)    ← NEW
Phase 5   SEO metadata
Phase 6   Voiceover               (Kokoro local TTS)
Phase 7   Music selection         (emotion-aware)
Phase 8   Video assembly          (Ken Burns · transitions · subtitles)
```

## Quick start

```bash
# 1. Clone or unzip the project
cd bot_v4

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env — add GROQ_API_KEY and at least one stock media key

# 5. Run
python main.py             # Reddit Drama (default)
python main.py --niche C   # AI & Tech News
```

## API keys

| Key | Required | Free tier | Link |
|-----|----------|-----------|------|
| `GROQ_API_KEY` | Yes | Yes | [console.groq.com](https://console.groq.com) |
| `PIXABAY_API_KEY` | One of these | Yes | [pixabay.com/api/docs](https://pixabay.com/api/docs/) |
| `PEXELS_API_KEY` | One of these | Yes | [pexels.com/api](https://www.pexels.com/api/) |
| `UNSPLASH_ACCESS_KEY` | Optional | Yes | [unsplash.com/developers](https://unsplash.com/developers) |

Groq is used for: script generation, scene planning, shot planning, story brain scoring, AI query improvement. All on the free tier.

The visual verifier uses `llama-3.2-11b-vision-preview` (also free on Groq).

## Niches

| Flag | Niche | Source |
|------|-------|--------|
| `A` | Reddit Drama / AITA | r/AmItheAsshole, r/relationship_advice, etc. |
| `B` | Science & Nature | Science APIs / curated feeds |
| `C` | AI & Tech News | Tech news feeds |
| `D` | History & Mystery | History / mystery feeds |

## v4 Feature flags (`.env`)

### Story Brain
```
STORY_BRAIN_ENABLED=true         # master switch
STORY_BRAIN_THRESHOLD=42         # score below this = weak asset (0-100)
STORY_BRAIN_MAX_RETRIES=2        # re-fetch attempts per weak scene
```
The Story Brain uses the full story arc to generate much more targeted queries for re-fetches. It accepts the improvement only if the new asset scores higher than the original.

### Shot Planner
```
SHOT_PLANNING_ENABLED=true
SHOT_PLANNING_MAX=4              # max shots per scene
```
Shot sequences by beat:
- `hook` → wide + close  
- `build_up` → wide + close  
- `discovery` → wide + close + reaction  
- `climax` → wide + close + reaction + detail  
- `resolution` → close + wide  

Very short scenes (< 3.6s) are not split.

### Sequencer
```
SEQUENCER_ENABLED=true
SEQUENCER_VIDEO_CADENCE=2        # every Nth shot = video
SEQUENCER_CLIMAX_SANDWICH=true   # image → climax_video → image
```
Climax beats always get video. Hook and resolution prefer images. No two consecutive videos (the weaker emotion is demoted to image).

### Visual Verifier
```
VISUAL_VERIFY_ENABLED=true
VISUAL_VERIFY_THRESHOLD=35       # score below this = rejected
```
Uses `llama-3.2-11b-vision-preview` (free, ~1-2s per scene). For videos, the first frame is extracted as the inspection image. Falls back gracefully if the vision call fails.

## File structure

```
bot_v4/
├── main.py                ← orchestrates the full pipeline
├── config.py              ← all settings and feature flags
├── .env.example           ← copy to .env and fill in keys
├── requirements.txt
│
├── content_fetcher.py     ← Phase 1: fetch Reddit / news content
├── script_generator.py    ← Phase 2+5: write script, titles, SEO
├── scene_planner.py       ← Phase 3: LLM director + AI queries (upgraded v4)
├── shot_planner.py        ← Phase 3b: ★ NEW — multi-shot scene expansion
├── sequencer.py           ← Phase 3c+4c: ★ NEW — image-video rhythm
├── asset_fetcher.py       ← Phase 4: fetch + vision verify (upgraded v4)
├── story_brain.py         ← Phase 4b: ★ NEW — LLM feedback loop
├── visual_verifier.py     ← used by asset_fetcher: ★ NEW — vision check
├── asset_ranker.py        ← heuristic + optional LLM ranking
├── image_generator.py     ← Pollinations.ai fallback image generation
├── character_registry.py  ← Phase 12: character consistency across scenes
├── voiceover.py           ← Phase 6: Kokoro local TTS
├── music_selector.py      ← Phase 7: emotion-aware music selection
├── video_maker.py         ← Phase 8: video assembly (Ken Burns, transitions)
├── subtitle_maker.py      ← Phase 11: subtitle overlay
├── background_fetcher.py  ← background video support
├── scene_builder.py       ← scene clip building utilities
├── niche_config.py        ← niche definitions and settings
└── scraper.py             ← web scraping utilities
```

## Disabling v4 features

Set any flag to `false` in `.env` to fall back to v3 behaviour:

```
STORY_BRAIN_ENABLED=false
SHOT_PLANNING_ENABLED=false
SEQUENCER_ENABLED=false
VISUAL_VERIFY_ENABLED=false
```

All four can be independently toggled. The pipeline is graceful: if any v4 phase fails with an exception, it logs a warning and continues with the previous result.

## Output

Each run produces:
- `output/<post_id>_video.mp4` — the final Short
- `output/<post_id>_audio.mp3` — voiceover only
- `output/<post_id>_meta.json` — full metadata including v4 story brain scores
- `output/scene_assets/` — all downloaded and generated scene images/videos

The metadata JSON includes a `v4_stats` block:
```json
{
  "v4_stats": {
    "original_scenes": 8,
    "expanded_shots": 22,
    "story_brain_scores": [78, 82, 45, 91, ...],
    "story_brain_avg": 74.3,
    "vision_scores": [85, 90, 60, 95, ...],
    "vision_avg": 82.5,
    "fallback_count": 1
  }
}
```
