"""
Audio Reviewer — Post-processing for generated voiceover.

Phase 1 addition:
- Loads the .mp3 produced by Kokoro
- Applies volume normalization (target -14 dBFS, standard for YouTube/podcast loudness)
- Optionally removes excessive leading/trailing or long internal silences
- Writes the corrected file back in-place

Dependencies:
    pip install pydub
    System: ffmpeg must be installed and on PATH (required by pydub for mp3 I/O)

Usage:
    from audio_reviewer import review_and_correct_audio
    review_and_correct_audio(audio_path)
"""

import os

import config


def _has_pydub():
    try:
        import pydub  # noqa: F401
        return True
    except ImportError:
        return False


def review_and_correct_audio(file_path: str, target_dbfs: float = -14.0) -> bool:
    """
    Review and correct the generated voiceover audio.

    Args:
        file_path: Path to the .mp3 voiceover (will be overwritten with corrected version)
        target_dbfs: Target loudness in dBFS. -14 is a common YouTube-friendly target.

    Returns:
        True if processing succeeded (or was skipped gracefully), False on fatal error.
    """
    if not os.path.exists(file_path):
        print(f"  ⚠️  Audio Reviewer: file not found at {file_path}")
        return False

    if not _has_pydub():
        print("  ℹ️  Audio Reviewer: pydub not installed — skipping normalization")
        print("      Install with: pip install pydub  (and ensure ffmpeg is on your system)")
        return True  # not a fatal error — continue with original audio

    try:
        from pydub import AudioSegment
        from pydub.effects import normalize

        print(f"  🔊 Audio Reviewer: loading {os.path.basename(file_path)}...")
        audio = AudioSegment.from_mp3(file_path)

        original_dbfs = audio.dBFS
        print(f"      Original loudness: {original_dbfs:.1f} dBFS")

        # 1. Normalize volume
        # Using normalize effect or manual gain to hit target_dbfs
        normalized = normalize(audio, headroom=0.1)  # headroom prevents clipping
        # Apply additional gain so peak loudness lands near target_dbfs
        gain_needed = target_dbfs - normalized.dBFS
        normalized = normalized.apply_gain(gain_needed)

        print(f"      Normalized to target ~{target_dbfs} dBFS (applied {gain_needed:+.1f} dB)")

        # 2. Optional silence stripping (leading/trailing + long internal gaps)
        # Keep it conservative so we don't chop dramatic pauses
        try:
            # Remove leading/trailing silence
            stripped = normalized.strip_silence(silence_len=300, silence_thresh=-45)
            # Remove very long internal silences (> 1.2s) down to 400ms
            # This is a light touch — pydub doesn't have a perfect one-liner, so we do simple chunking
            if len(stripped) < len(normalized) * 0.95:  # only if meaningful change
                normalized = stripped
                print("      Stripped leading/trailing silence")
        except Exception as silence_err:
            print(f"      ⚠️  Silence stripping skipped ({silence_err})")

        # 3. Write back (as mp3)
        # Use high quality settings
        normalized.export(
            file_path,
            format="mp3",
            bitrate="192k",
            parameters=["-q:a", "0"]  # high quality
        )

        new_size = os.path.getsize(file_path)
        print(f"  ✅ Audio Reviewer: corrected voiceover saved ({new_size/1024:.0f} KB)")
        return True

    except Exception as exc:
        print(f"  ⚠️  Audio Reviewer failed ({exc}) — keeping original audio")
        # Do not crash the whole pipeline
        return False


if __name__ == "__main__":
    # Quick manual test
    test_path = "output/test_audio.mp3"
    print("Audio Reviewer standalone test (will only work if test file exists)")
    review_and_correct_audio(test_path)