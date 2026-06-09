import soundfile as sf
import numpy as np
import os

# ── Voice selection ────────────────────────────────────────
# American English voices:
#   Female: af_heart, af_bella, af_sarah, af_sky, af_nicole
#   Male:   am_adam, am_michael
# British English voices:
#   Female: bf_emma, bf_isabella
#   Male:   bm_george, bm_lewis

VOICE       = "am_adam"   # Deep male narrator
SPEED       = 0.85        # Slightly slower = more dramatic
MODEL_PATH  = "assets/kokoro-v1.0.onnx"
VOICES_PATH = "assets/voices-v1.0.bin"


def split_text(text, max_chars=200):
    """Split text into chunks under max_chars at sentence boundaries."""
    sentences = text.replace("...", "…").split(". ")
    chunks    = []
    current   = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        candidate = current + (". " if current else "") + sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current + ".")
            current = sentence

    if current:
        chunks.append(current if current.endswith(".") else current + ".")

    return chunks


def generate_voiceover(script, output_path):
    """Convert script to MP3 using Kokoro-ONNX (local, free, Python 3.13 compatible)."""
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        raise ImportError(
            "kokoro-onnx not installed. Run: pip install kokoro-onnx soundfile"
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Download it with:\n"
            "curl -L -o assets/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        )

    if not os.path.exists(VOICES_PATH):
        raise FileNotFoundError(
            f"Voices not found at {VOICES_PATH}.\n"
            "Download it with:\n"
            "curl -L -o assets/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        )

    print(f"  Generating voiceover with Kokoro-ONNX ({len(script)} chars)...")
    print(f"  Voice: {VOICE} | Speed: {SPEED}")

    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

    # Split into chunks for best quality
    chunks = split_text(script, max_chars=200)
    print(f"  Processing {len(chunks)} text chunks...")

    all_audio   = []
    sample_rate = 24000

    for i, chunk in enumerate(chunks):
        samples, rate = kokoro.create(chunk, voice=VOICE, speed=SPEED, lang="en-us")
        all_audio.append(samples)
        sample_rate = rate

    if not all_audio:
        raise RuntimeError("Kokoro produced no audio output.")

    # Concatenate chunks
    combined = np.concatenate(all_audio)

    # Save as WAV then convert to MP3 via ffmpeg
    wav_path = output_path.replace(".mp3", ".wav")
    sf.write(wav_path, combined, sample_rate)

    os.system(f'ffmpeg -y -i "{wav_path}" -codec:a libmp3lame -q:a 2 "{output_path}" -loglevel quiet')
    os.remove(wav_path)

    print(f"  ✅ Voiceover saved → {output_path}")
    return output_path
