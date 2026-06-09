"""
YouTube Shorts Bot — Main Pipeline (v5 + Phase 1-4 extensions)
Story-to-video engine with multimodal Story Brain + Audio Reviewer,
NASA/Wikimedia assets, Bensound music, and Grok-first image fallbacks.

Usage:
    python main.py --niche A        # Reddit Drama (default)
    python main.py --niche B        # Science / Nature
    python main.py --niche C        # AI & Tech News
    python main.py --niche D        # History & Mystery

Phase improvements over v3:
  ★ Story Brain       — LLM feedback loop: scores every fetched asset, re-fetches weak ones
  ★ Shot Planning     — expands each scene into wide + close-up + reaction + detail shots
  ★ Sequencer (pre)   — plans image→video→image rhythm BEFORE fetch (better first-pass results)
  ★ Visual Verifier   — vision model checks every downloaded asset actually matches the scene
  ★ Sequencer (post)  — enforces the planned rhythm AFTER fetch (catches fallbacks in video slots)
  ★ AI Search Queries — LLM generates concrete stock queries from story arc, not keyword rules

Phase improvements over v4 (NEW in v5):
  ★ CLIP Retrieval        — semantic embedding ranking; "grief alone" matches "female devastation isolated"
  ★ Story Brain Vision    — Story Brain SEES the actual image (not just metadata tags)
  ★ Editing Intelligence  — beat-aware cut decisions: smash cuts, reaction-first, hold frames, Ken Burns
  ★ Multimodal Memory     — Story Brain remembers characters/locations/timeline across all scenes
  ★ Character Embeddings  — embedding-based face consistency (CLIP/MiniLM/InsightFace auto-selected)

Phases executed per run:
  1   Content fetching           content_fetcher.py
  2   Script generation          script_generator.py
  3   Scene planning             scene_planner.py       (LLM director + AI queries)
  3b  Shot planning              shot_planner.py        (★ NEW: multi-shot expansion)
  3c  Pre-fetch sequencing       sequencer.py           (★ NEW: image-video plan)
  4   Asset fetching             asset_fetcher.py       (CLIP semantic ranking ★ v5)
  4b  Story Brain review         story_brain.py         (★ v5: now SEES images + multimodal memory)
  4c  Post-fetch sequencing      sequencer.py           (★ enforce rhythm after fetch)
  4d  Editing Intelligence       editing_intelligence.py (★ v5 NEW: cut decisions, pacing, Ken Burns)
  5   Titles / SEO metadata      script_generator.py
  6   Voiceover                  voiceover.py           (Kokoro TTS, local)
  7   Music selection            music_selector.py      (emotion-aware)
  8   Video assembly             video_maker.py         (Ken Burns, transitions)
  9   Output & checklist
"""

import argparse
import json
import os
import sys
import traceback

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import config
from character_registry  import CharacterRegistry
from content_fetcher     import fetch_content
from music_selector      import get_music_for_scene_plan
from niche_config        import get_niche
from scene_planner       import generate_scene_plan
from shot_planner        import expand_scene_plan
from sequencer           import apply_sequence_to_plan, resequence_fetched_assets
from asset_fetcher       import fetch_assets
from story_brain         import review_and_improve
from editing_intelligence import annotate_edit_decisions   # v5
from script_generator    import generate_script, generate_title_and_description
from video_maker         import create_video
from voiceover           import generate_voiceover
from audio_reviewer      import review_and_correct_audio   # Phase 1: post-voiceover normalization

# ── Dirs ──────────────────────────────────────────────────────────────────────
for _d in (config.OUTPUT_DIR, config.SCENE_ASSET_DIR, "assets"):
    os.makedirs(_d, exist_ok=True)


# ── Duplicate guard ───────────────────────────────────────────────────────────

def _load_used() -> set:
    if not os.path.exists(config.USED_IDS_PATH):
        return set()
    with open(config.USED_IDS_PATH) as f:
        return {l.strip() for l in f if l.strip()}


def _save_used(post_id: str) -> None:
    with open(config.USED_IDS_PATH, "a") as f:
        f.write(post_id + "\n")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(niche_key: str = "A") -> bool:
    """
    Run the full 20-phase pipeline for one video.
    Returns True on success, False on any fatal error.
    """
    niche = get_niche(niche_key)
    sep   = "═" * 60

    print(f"\n{sep}")
    print(f"  🎬 YouTube Shorts Bot v4  [{niche['emoji']} {niche['name']}]")
    print(f"  ★ Story Brain · Shot Planning · Sequencer · Vision Verify")
    print(f"{sep}\n")

    # ── Phase 1: Content ──────────────────────────────────
    _step(1, "Fetching content")
    used = _load_used()
    post = fetch_content(niche_key, used)
    if not post:
        print("  ❌ No suitable content found — try again later")
        return False
    print(f"  ✅ \"{post['title'][:70]}\"")
    print(f"     Source: {post.get('subreddit', post.get('source', '?'))}")

    # ── Phase 2: Script ───────────────────────────────────
    _step(2, "Writing script")
    try:
        script, mood = generate_script(post, niche=niche_key)
        words = len(script.split())
        print(f"  ✅ {words} words | mood: {mood}")
    except Exception as e:
        _fatal(2, e); return False

    # ── Phase 3: Scene plan (Director + AI queries) ───────
    _step(3, "Director building scene plan  (LLM + story-aware queries)")
    try:
        scene_plan = generate_scene_plan(
            script,
            post=post,
            niche=niche_key,
            target_duration=config.TARGET_SECONDS,
        )
        print(f"  ✅ {len(scene_plan)} scenes planned")
        for i, s in enumerate(scene_plan, 1):
            print(f"     [{i}] {s['beat']:12s} | {s['emotion']:10s} | "
                  f"{s['duration']}s | {s['transition']} | "
                  f"\"{s['visual_goal'][:45]}\"")
    except Exception as e:
        _fatal(3, e); return False

    # ── Phase 3b: Shot Planning (★ NEW) ───────────────────
    _step("3b", "Shot planning  (★ multi-shot scene expansion)")
    try:
        if config.SHOT_PLANNING_ENABLED:
            expanded_plan = expand_scene_plan(scene_plan)
            print(f"  ✅ {len(scene_plan)} scenes → {len(expanded_plan)} shots "
                  f"(wide + close-up + reaction + detail)")
        else:
            expanded_plan = scene_plan
            print("  ℹ️  Shot planning disabled (SHOT_PLANNING_ENABLED=false)")
    except Exception as e:
        print(f"  ⚠️  Shot planning failed ({e}) — continuing with original plan")
        expanded_plan = scene_plan

    # ── Phase 3c: Pre-fetch Sequencing (★ NEW) ────────────
    _step("3c", "Pre-fetch sequencing  (★ image→video→image rhythm plan)")
    try:
        if config.SEQUENCER_ENABLED:
            sequenced_plan = apply_sequence_to_plan(expanded_plan)
            print(f"  ✅ Sequence plan applied to {len(sequenced_plan)} shots")
        else:
            sequenced_plan = expanded_plan
            print("  ℹ️  Sequencer disabled (SEQUENCER_ENABLED=false)")
    except Exception as e:
        print(f"  ⚠️  Pre-fetch sequencing failed ({e}) — continuing without")
        sequenced_plan = expanded_plan

    # ── Phase 4: Asset fetching + ranking + vision verify ─
    _step(4, "Fetching scene assets  (multi-query · ranking · vision verify ★)")
    registry = CharacterRegistry()
    try:
        scene_assets = fetch_assets(sequenced_plan, character_registry=registry)
        fallbacks    = sum(1 for a in scene_assets if a.get("fallback"))
        stock        = len(scene_assets) - fallbacks
        print(f"  ✅ {stock} stock  |  {fallbacks} AI-generated fallback(s)")
    except Exception as e:
        _fatal(4, e); return False

    # ── Phase 4b: Story Brain review (★ NEW) ──────────────
    _step("4b", "Story Brain review  (★ LLM feedback loop — scoring & re-fetching weak assets)")
    try:
        if config.STORY_BRAIN_ENABLED:
            scene_assets = review_and_improve(
                sequenced_plan,
                scene_assets,
                character_registry=registry,
            )
            improved_count = sum(
                1 for a in scene_assets
                if a.get("story_score", 100) >= config.STORY_BRAIN_THRESHOLD
            )
            print(f"  ✅ Story Brain: {improved_count}/{len(scene_assets)} assets at or above threshold")
        else:
            print("  ℹ️  Story Brain disabled (STORY_BRAIN_ENABLED=false)")
    except Exception as e:
        print(f"  ⚠️  Story Brain failed ({e}) — continuing with original assets")

    # ── Phase 4c: Post-fetch Sequencing (★ NEW) ───────────
    _step("4c", "Post-fetch sequencing  (★ enforcing image→video rhythm after fetch)")
    try:
        if config.SEQUENCER_ENABLED:
            scene_assets = resequence_fetched_assets(
                sequenced_plan,
                scene_assets,
                character_registry=registry,
            )
            print(f"  ✅ Final sequence enforced")
        else:
            print("  ℹ️  Post-fetch sequencing skipped")
    except Exception as e:
        print(f"  ⚠️  Post-fetch sequencing failed ({e}) — continuing without")

    # ── Phase 4d: Editing Intelligence (★ v5 NEW) ─────────
    _step("4d", "Editing Intelligence  (★ v5: beat-aware cut decisions, pacing, Ken Burns)")
    try:
        if getattr(config, "EDITING_INTELLIGENCE_ENABLED", True):
            use_llm = getattr(config, "EDITING_INTELLIGENCE_LLM", True)
            scene_assets = annotate_edit_decisions(
                scene_assets, use_llm_for_climax=use_llm
            )
            smash_cuts = sum(
                1 for a in scene_assets
                if a.get("edit", {}).get("cut_style") == "smash"
            )
            print(f"  ✅ Edit decisions applied: {smash_cuts} smash cut(s) planned")
        else:
            print("  ℹ️  Editing Intelligence disabled (EDITING_INTELLIGENCE_ENABLED=false)")
    except Exception as e:
        print(f"  ⚠️  Editing Intelligence failed ({e}) — continuing without")

    # ── Phase 5: Titles / SEO ─────────────────────────────
    _step(5, "Generating titles, description, hashtags")
    try:
        meta = generate_title_and_description(post, script, niche=niche_key, mood=mood)
    except Exception as e:
        print(f"  ⚠️  Metadata generation failed ({e}) — using defaults")
        meta = {
            "titles":         [post["title"]],
            "description":    script[:200] + "…",
            "hashtags":       " ".join(niche["hashtags_base"]),
            "pinned_comment": "What would you do? 👇",
        }
    print(f"  ✅ {len(meta['titles'])} title(s) generated")

    # ── Phase 6: Voiceover (Kokoro TTS) ───────────────────
    _step(6, "Generating voiceover  (Kokoro local TTS)")
    audio_path = f"{config.OUTPUT_DIR}/{post['id']}_audio.mp3"
    try:
        generate_voiceover(script, audio_path)
    except Exception as e:
        _fatal(6, e); return False

    # ── Phase 6b: Audio Reviewer (NEW) ────────────────────
    _step("6b", "Reviewing & normalizing audio  (volume, silence)")
    try:
        review_and_correct_audio(audio_path)
    except Exception as e:
        print(f"  ⚠️  Audio review failed ({e}) — continuing with original")

    # ── Phase 7 / 10: Emotion-aware music ─────────────────
    _step("7+10", "Selecting music  (emotion-aware)")
    # Use original scene_plan for music mood (before shot expansion)
    music_path = get_music_for_scene_plan(scene_plan, niche_mood=niche["music_mood"] or mood)

    # ── Phase 8/9/11: Video assembly ──────────────────────
    _step("8·9·11", "Assembling video  (Ken Burns · transitions · subtitles)")
    video_path = f"{config.OUTPUT_DIR}/{post['id']}_video.mp4"
    try:
        create_video(
            audio_path  = audio_path,
            output_path = video_path,
            script      = script,
            music_path  = music_path,
            scene_assets= scene_assets,
        )
    except Exception as e:
        _fatal("8-11", e)
        video_path = None

    # ── Save metadata JSON ─────────────────────────────────
    meta_path = f"{config.OUTPUT_DIR}/{post['id']}_meta.json"
    _save_metadata(meta_path, post, niche_key, niche, meta, mood,
                   script, scene_plan, expanded_plan, scene_assets,
                   audio_path, music_path, video_path)
    _save_used(post["id"])

    # ── Upload checklist ───────────────────────────────────
    _print_checklist(niche, meta, video_path, audio_path, meta_path,
                     scene_plan, scene_assets)
    return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _step(n, label: str) -> None:
    print(f"\n── Phase {n}: {label}")


def _fatal(n, exc: Exception) -> None:
    print(f"  ❌ Phase {n} failed: {exc}")
    traceback.print_exc()


def _save_metadata(path, post, niche_key, niche, meta, mood,
                   script, scene_plan, expanded_plan, scene_assets,
                   audio_path, music_path, video_path):
    # Compute story brain stats for metadata
    story_scores = [a.get("story_score") for a in scene_assets if a.get("story_score") is not None]
    vision_scores = [a.get("vision_score") for a in scene_assets if a.get("vision_score") is not None]

    data = {
        "version": "4.0",
        "niche": niche_key, "niche_name": niche["name"],
        "source_id": post["id"], "source_url": post.get("url", ""),
        "titles": meta["titles"], "description": meta["description"],
        "hashtags": meta["hashtags"], "pinned_comment": meta["pinned_comment"],
        "mood": mood, "script": script,
        "scene_plan": scene_plan,
        "expanded_shot_plan": expanded_plan,
        "scene_assets": [
            {k: v for k, v in a.items() if k != "asset"}   # omit local path
            for a in scene_assets
        ],
        "v4_stats": {
            "original_scenes": len(scene_plan),
            "expanded_shots": len(expanded_plan),
            "story_brain_scores": story_scores,
            "story_brain_avg": round(sum(story_scores) / len(story_scores), 1) if story_scores else None,
            "vision_scores": vision_scores,
            "vision_avg": round(sum(vision_scores) / len(vision_scores), 1) if vision_scores else None,
            "fallback_count": sum(1 for a in scene_assets if a.get("fallback")),
        },
        "files": {
            "audio": audio_path,
            "music": music_path,
            "video": video_path,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_checklist(niche, meta, video_path, audio_path, meta_path,
                     scene_plan, scene_assets):
    sep = "═" * 60
    print(f"\n{sep}")
    if video_path and os.path.exists(video_path):
        print(f"  🎉 {niche['emoji']}  SHORT READY  (v4)")
    else:
        print(f"  ❌ {niche['emoji']}  VIDEO GENERATION FAILED  (v4)")
    print(f"{sep}")
    if video_path and os.path.exists(video_path):
        mb = os.path.getsize(video_path) / 1024 / 1024
        print(f"\n  📹 Video  → {video_path}  ({mb:.1f} MB)")
    print(f"  🎵 Audio  → {audio_path}")
    print(f"  📄 Meta   → {meta_path}")

    # Scene breakdown with story brain scores
    print(f"\n  🎬 SCENE BREAKDOWN (with v4 quality scores)")
    for i, (s, a) in enumerate(zip(scene_plan, scene_assets[:len(scene_plan)]), 1):
        story_score = a.get("story_score", "—")
        vision_score = a.get("vision_score", "—")
        score_str = f"story={story_score} vision={vision_score}"
        print(f"     [{i}] {s['beat']:11s} | {s['emotion']:10s} | "
              f"{s.get('transition','cut'):10s} | {s['duration']}s | {score_str}")

    # Story brain summary
    story_scores = [a.get("story_score") for a in scene_assets if a.get("story_score") is not None]
    if story_scores:
        avg = sum(story_scores) / len(story_scores)
        weak = sum(1 for sc in story_scores if sc < config.STORY_BRAIN_THRESHOLD)
        print(f"\n  🧠 Story Brain avg score: {avg:.1f}/100  |  {weak} weak scene(s) found")

    # Upload checklist
    print(f"\n  📋 UPLOAD CHECKLIST  (copy-paste ready)")
    print(f"  {'─'*52}")
    for i, t in enumerate(meta["titles"], 1):
        print(f"  Title {i}: {t}")
    print(f"\n  Description:")
    for line in meta["description"].splitlines():
        print(f"    {line}")
    print(f"\n  Hashtags:  {meta['hashtags']}")
    print(f"\n  📌 Pin after upload:")
    print(f"     \"{meta['pinned_comment']}\"")
    print(f"\n  ⏰ Best upload times: 7–9 PM IST  or  12–2 PM IST")
    print(f"{sep}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YouTube Shorts Bot v4 — story-to-video with Story Brain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Niches:
  A  Reddit Drama / AITA / Relationships  (default)
  B  Science & Nature Facts
  C  AI & Tech News
  D  History & Mystery

Examples:
  python main.py
  python main.py --niche C
  python main.py -n D

v4 Feature flags (set in .env or environment):
  STORY_BRAIN_ENABLED=true        LLM feedback loop (default: on)
  STORY_BRAIN_THRESHOLD=42        Score below this = weak (0-100)
  STORY_BRAIN_MAX_RETRIES=2       Re-fetch attempts per weak scene
  SHOT_PLANNING_ENABLED=true      Multi-shot expansion (default: on)
  SHOT_PLANNING_MAX=4             Max shots per scene
  SEQUENCER_ENABLED=true          Image-video rhythm (default: on)
  SEQUENCER_VIDEO_CADENCE=2       Every Nth shot = video
  VISUAL_VERIFY_ENABLED=true      Vision model check (default: on)
  VISUAL_VERIFY_THRESHOLD=35      Score below this = rejected (0-100)
        """,
    )
    parser.add_argument("--niche", "-n", default="A",
                        choices=["A","B","C","D","a","b","c","d"],
                        metavar="NICHE")
    args = parser.parse_args()
    ok   = run_pipeline(args.niche.upper())
    sys.exit(0 if ok else 1)
