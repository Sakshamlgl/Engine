"""
Media Sequencer — Image-Video-Image Alternating Pattern

Transforms the current flat sequence:

    image → image → video → image → image → video

into a deliberate alternating rhythm:

    image → video → image → video → image → video

This creates a more dynamic, professional feel. Static images provide
emotional anchor points; motion clips maintain energy and attention.

The sequencer works as a POST-FETCH step: after assets are fetched,
it reviews the sequence and swaps asset_types where needed to enforce
the alternating pattern, then triggers re-fetches for swapped slots.

It also handles the "video sandwich" pattern for climax scenes:
    image (build) → video (action/reveal) → image (reaction/freeze)

This is particularly effective for shock and climax beats.

Configuration (config.py / .env):
    SEQUENCER_ENABLED         = true
    SEQUENCER_VIDEO_CADENCE   = 2     # every Nth scene gets a video (default: 2)
    SEQUENCER_CLIMAX_SANDWICH = true  # wrap climax videos in images
"""

import config

SEQUENCER_ENABLED         = getattr(config, "SEQUENCER_ENABLED",         True)
SEQUENCER_VIDEO_CADENCE   = getattr(config, "SEQUENCER_VIDEO_CADENCE",   2)
SEQUENCER_CLIMAX_SANDWICH = getattr(config, "SEQUENCER_CLIMAX_SANDWICH", True)


def _wants_video(index: int, scene: dict, target_indices: set) -> bool:
    """Return True if this scene should be a video slot."""
    return index in target_indices


def _compute_video_slots(scenes: list) -> set:
    """
    Compute which scene indices should be video slots.

    Rules:
    1. Every SEQUENCER_VIDEO_CADENCE-th scene gets a video (alternating pattern)
    2. climax beats always get a video
    3. hook scenes prefer images (first impression = stable frame)
    4. resolution scenes prefer images (emotional close)
    5. Never two consecutive videos
    """
    n = len(scenes)
    video_slots = set()

    # Base alternating pattern: every Nth scene
    for i in range(SEQUENCER_VIDEO_CADENCE - 1, n, SEQUENCER_VIDEO_CADENCE):
        video_slots.add(i)

    # Force climax to video
    for i, scene in enumerate(scenes):
        if scene.get("beat") == "climax":
            video_slots.add(i)

    # Remove consecutive videos (keep the one with higher emotion weight)
    emotion_weight = {
        "shock": 5, "dramatic": 4, "suspense": 3, "emotional": 3,
        "uplifting": 2, "mysterious": 2, "sadness": 2, "funny": 1,
    }
    to_remove = set()
    sorted_slots = sorted(video_slots)
    for i in range(len(sorted_slots) - 1):
        a, b = sorted_slots[i], sorted_slots[i + 1]
        if b - a == 1:
            # Consecutive — remove the weaker one
            w_a = emotion_weight.get(scenes[a].get("emotion", ""), 1)
            w_b = emotion_weight.get(scenes[b].get("emotion", ""), 1)
            if w_a >= w_b:
                to_remove.add(b)
            else:
                to_remove.add(a)
    video_slots -= to_remove

    # Hook and resolution prefer images
    for i, scene in enumerate(scenes):
        if scene.get("beat") in ("hook", "resolution") and i in video_slots:
            # Only remove if there's another video nearby
            nearby_video = any(
                j in video_slots for j in range(max(0, i - 2), min(n, i + 3))
                if j != i
            )
            if nearby_video:
                video_slots.discard(i)

    return video_slots


def _climax_sandwich_indices(scenes: list, video_slots: set) -> tuple[set, set]:
    """
    For climax video scenes, identify the scene before and after as image "bread".
    Returns (pre_image_indices, post_image_indices).
    """
    if not SEQUENCER_CLIMAX_SANDWICH:
        return set(), set()

    n = len(scenes)
    pre_image  = set()
    post_image = set()

    for i, scene in enumerate(scenes):
        if scene.get("beat") == "climax" and i in video_slots:
            if i > 0:
                pre_image.add(i - 1)
            if i < n - 1:
                post_image.add(i + 1)

    # These must be images — remove from video_slots if needed
    video_slots -= pre_image
    video_slots -= post_image

    return pre_image, post_image


def plan_sequence(scenes: list) -> list[str]:
    """
    Return a list of intended media types ("image" or "video") for each scene.

    This is the sequence plan — it doesn't fetch anything, just decides
    which scenes should be images vs videos.
    """
    n = len(scenes)
    if n == 0:
        return []

    video_slots = _compute_video_slots(scenes)
    pre_img, post_img = _climax_sandwich_indices(scenes, video_slots)

    sequence = []
    for i in range(n):
        if i in video_slots:
            sequence.append("video")
        else:
            sequence.append("image")

    return sequence


def apply_sequence_to_plan(scene_plan: list) -> list:
    """
    Update each scene's asset_types to enforce the image-video-image sequence.
    Call this BEFORE fetch_assets so the fetcher searches for the right media type.
    """
    if not SEQUENCER_ENABLED or not scene_plan:
        return scene_plan

    sequence = plan_sequence(scene_plan)
    updated  = []

    for scene, intended_type in zip(scene_plan, sequence):
        current_types = scene.get("asset_types", ["image", "video"])
        if intended_type == "video":
            new_types = ["video", "image"]
        else:
            new_types = ["image", "video"]

        updated.append({**scene, "asset_types": new_types})

    # Log the sequence
    icons = {"video": "🎬", "image": "🖼️"}
    seq_str = " → ".join(icons.get(t, "?") for t in sequence)
    print(f"  🎞️  Sequence plan: {seq_str}")

    return updated


def resequence_fetched_assets(
    scene_plan: list,
    scene_assets: list,
    character_registry=None,
) -> list:
    """
    POST-FETCH sequencing pass.

    After assets are fetched, checks if the actual sequence matches the plan.
    If a video slot got an image (or vice versa), triggers a targeted re-fetch
    for that scene with the correct media type forced.

    This is the fallback for when the initial fetch didn't find the right type.
    """
    if not SEQUENCER_ENABLED or not scene_assets:
        return scene_assets

    from asset_fetcher import fetch_scene_asset

    sequence   = plan_sequence(scene_plan)
    resequenced = list(scene_assets)
    swaps      = 0

    for idx, (intended, asset, scene) in enumerate(zip(sequence, resequenced, scene_plan)):
        actual_type = asset.get("type", "image")

        # If it's a fallback/generated image in a video slot, try harder for video
        is_fallback = asset.get("fallback", False)
        type_mismatch = (intended == "video" and actual_type == "image" and is_fallback)

        if type_mismatch:
            print(f"  🎞️  Scene {idx + 1}: sequence wants {intended}, got {actual_type} (fallback) — retrying")
            patched = {**scene, "asset_types": [intended, "image" if intended == "video" else "video"]}
            try:
                new_asset = fetch_scene_asset(
                    patched,
                    idx + 1,
                    previous_assets=resequenced[:idx],
                    character_registry=character_registry,
                )
                if new_asset.get("type") == intended and not new_asset.get("fallback"):
                    resequenced[idx] = new_asset
                    swaps += 1
                    print(f"    ✅ re-sequenced to {intended}")
            except Exception as exc:
                print(f"    re-sequence fetch failed ({exc}), keeping original")

    if swaps:
        print(f"  🎞️  Sequencer: {swaps} slot(s) re-fetched for better media type")

    # Log final sequence
    icons = {"video": "🎬", "image": "🖼️"}
    final_seq = " → ".join(icons.get(a.get("type", "image"), "?") for a in resequenced)
    print(f"  🎞️  Final sequence: {final_seq}")

    return resequenced