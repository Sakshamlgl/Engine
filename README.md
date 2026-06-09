# Automated Short-Form Video Pipeline

An AI-powered content generation system that transforms Reddit stories into publish-ready YouTube Shorts through automated script generation, multimodal scene planning, semantic asset retrieval, narration synthesis, and video assembly.

## Features

- End-to-end Reddit-to-YouTube Shorts automation
- Multimodal Story Brain with iterative asset refinement
- CLIP-powered semantic visual retrieval
- Vision-language asset verification
- Beat-aware cinematic editing intelligence
- Character consistency tracking
- Automated subtitle generation
- Automated SEO metadata generation
- Fully automated video assembly pipeline

## Architecture

```text
Reddit Content Fetching
          ↓
Script Generation
          ↓
Scene Planning
          ↓
Story Brain Review
          ↓
Asset Retrieval
          ↓
Visual Verification
          ↓
Asset Re-ranking
          ↓
Voice Synthesis
          ↓
Subtitle Generation
          ↓
Video Assembly
          ↓
SEO Metadata Generation
```

## Tech Stack

### AI / ML
- Groq LLMs
- Vision-Language Models
- CLIP Embeddings
- MiniLM
- InsightFace

### Backend
- Python
- PRAW
- FFmpeg
- MoviePy

### Media Processing
- Kokoro TTS
- OpenCV
- Subtitle Automation

## Project Structure

```text
bot_v5/
├── main.py
├── story_brain.py
├── visual_verifier.py
├── clip_ranker.py
├── asset_ranker.py
├── editing_intelligence.py
├── script_generator.py
├── voiceover.py
├── subtitle_maker.py
├── sequencer.py
└── ...
```

## Installation

```bash
git clone https://github.com/yourusername/automated-shortform-video-pipeline.git
cd automated-shortform-video-pipeline

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
python main.py --niche A
```

## Highlights

- 23+ modular Python components
- Automated Reddit-to-video workflow
- Semantic asset retrieval using CLIP embeddings
- Context-aware visual validation and re-fetching
- Narrative-aware editing intelligence
- Character consistency across scenes
- Zero paid API dependency for identity tracking

## Future Roadmap

- Image-to-video generation
- Multi-agent story planning
- Persistent cross-video memory
- Multi-language support
- Advanced visual consistency models

## Resume Summary

Built a production-scale AI video generation pipeline that converts Reddit stories into publish-ready YouTube Shorts using multimodal reasoning, semantic retrieval, automated narration, visual verification, and intelligent video editing.
