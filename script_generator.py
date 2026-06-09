from groq import Groq
import config

client = Groq(api_key=config.GROQ_API_KEY)

# ── Per-niche system prompts ───────────────────────────────

SYSTEM_PROMPTS = {
    "A": """You are a viral YouTube Shorts script writer for Reddit story narration.
You write punchy, dramatic third-person narrations that stop the scroll in 2 seconds.
Never say "OP", "Original Poster", or "I". No stage directions or headers.
Every word earns its place — no filler, no slow build-up.
CRITICAL: Do NOT editorialize, exaggerate, or change the core facts of the source post. Stay strictly true to the original text and its exact tone. Avoid injecting external societal commentary (e.g., toxic masculinity, predators) unless explicitly mentioned in the source.""",

    "B": """You are a viral YouTube Shorts script writer for mind-blowing science facts.
You make complex science feel urgent, personal, and jaw-dropping to a general audience.
You use vivid analogies and build to a reveal that makes people say "wait, WHAT?".
No jargon. No stage directions. Just pure awe-inspiring narration.""",

    "C": """You are a viral YouTube Shorts script writer for AI and tech news.
You make AI developments feel urgent, consequential, and easy to understand.
You explain what happened, why it's huge, and what it means for regular people.
Tone: confident, fast-paced, slightly alarmed. No jargon without instant explanation.""",

    "D": """You are a viral YouTube Shorts script writer for history mysteries.
You narrate forgotten, mysterious, or shocking historical events like a detective uncovering secrets.
Build dread, wonder, or disbelief. Make the listener feel like they've stumbled onto something hidden.
No academic tone. Pure storytelling mystery.""",
}

# ── Per-niche hook formulas ────────────────────────────────

HOOK_TEMPLATES = {
    "A": """
- Reveal the ending first: "She caught her husband of 12 years living a double life — and the proof was hiding in plain sight."
- Impossible question: "What would you do if your best friend showed up at your wedding... as the other bride?"
- Drop into action: "The moment she opened that text, her entire life fell apart."
- Create mystery: "He seemed like the perfect boyfriend — until she found the basement." """,

    "B": """
- Scale shock: "There are more neurons in your brain than stars in the Milky Way — and scientists just found out they can rewrite themselves."
- Counterintuitive: "The thing keeping you alive right now shouldn't be possible according to the laws of physics."
- Discovery reveal: "Scientists just found something inside a black hole that breaks every model we had."
- Personal stakes: "This discovery means everything you thought you knew about [X] is wrong." """,

    "C": """
- Urgency drop: "This AI just did something no human has ever done — and it happened yesterday."
- Stakes reveal: "OpenAI/Anthropic/Google just crossed a line that experts said was years away."
- Consequence hook: "This one AI update is about to change how [millions of people] do [thing] forever."
- Alarm: "The AI model that just launched can already do what was supposed to take a decade." """,

    "D": """
- Cold open mystery: "In 1872, they found a ship in the middle of the ocean — engine running, food still warm, crew completely gone."
- Forbidden knowledge: "Historians know this happened. They just refuse to explain how."
- Scale of weirdness: "This ancient structure is so precisely built that we still can't replicate it today."
- Conspiracy of silence: "For 200 years, governments actively hid this from history books." """,
}

# ── Per-niche ending lines ─────────────────────────────────

ENDINGS = {
    "A": "Drop your verdict below 👇 Who was wrong here?",
    "B": "Follow for more science facts that will break your brain 🧠",
    "C": "Follow to stay ahead of AI — this is only the beginning 🤖",
    "D": "Follow for more history mysteries they don't teach in school 🏛️",
}


def generate_script(post, niche="A"):
    """Generate a niche-appropriate 45-second Shorts script. Returns (script, mood)."""
    niche = niche.upper()

    prompt = f"""Turn this content into a VIRAL YouTube Shorts narration script.

Title: {post['title']}
Content: {post['content']}

STRICT RULES:
1. HOOK — first 1-2 sentences must be impossible to scroll past. Use one of:
{HOOK_TEMPLATES.get(niche, HOOK_TEMPLATES['A'])}

2. NARRATION:
{"- Third person only. Infer gender. Use name if given, else invent one. NEVER say 'OP' or 'I'." if niche == "A" else "- Narrate like a science communicator / news anchor / historian (match niche)."}
- Max 12 words per sentence.
- Every 2-3 sentences must reveal something NEW or raise stakes.
- Use "..." max 3 times for dramatic effect only.

3. PACING:
- No slow build-up — straight to the drama.
- EXACTLY 150-170 words total. Count before finishing.
- If under 140 words, add more detail or reaction.

4. ENDING — last line must be EXACTLY:
"{ENDINGS.get(niche, ENDINGS['A'])}"

After the script, on a NEW LINE write:
MOOD: <one of: dramatic, suspenseful, emotional, uplifting, funny>
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPTS.get(niche, SYSTEM_PROMPTS["A"])},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=700,
        temperature=0.88
    )

    raw  = response.choices[0].message.content.strip()
    mood = "dramatic"

    if "MOOD:" in raw:
        parts  = raw.rsplit("MOOD:", 1)
        script = parts[0].strip()
        mood   = parts[1].strip().lower().split()[0]
        if mood not in ("dramatic", "suspenseful", "emotional", "uplifting", "funny"):
            mood = "dramatic"
    else:
        script = raw

    word_count = len(script.split())
    print(f"  Script: {word_count} words | Mood: {mood}")
    return script, mood


# ── Title + Description + Hashtags ────────────────────────

TITLE_STYLES = {
    "A": """Curiosity-gap drama titles. Formats:
- "She [did X]... and it destroyed everything 😱"
- "He thought no one would find out. He was wrong. 🚨"
- "[X] years of [trust/love/loyalty]... gone in one night 💔"
- "The [person] who [shocking action] — and got away with it 😤" """,

    "B": """Mind-blowing science titles. Formats:
- "Scientists just discovered X — and it changes everything 🔬"
- "The [phenomenon] that shouldn't exist... but does 🤯"
- "Your [body/brain/universe] is doing something impossible right now 🧠"
- "We've been wrong about [X] for [N] years 🚀" """,

    "C": """Urgent AI news titles. Formats:
- "This AI just did X — the world is not ready 🤖"
- "[Company]'s new model can now [capability] — here's why that's huge ⚡"
- "The AI update that [millions] don't know about yet 🔥"
- "AI just crossed [milestone] — everything changes now 🌍" """,

    "D": """History mystery titles. Formats:
- "The [mystery] historians still can't explain 🏛️"
- "In [year], [shocking event] — and it was covered up for centuries 📜"
- "The [ancient thing] that proves [established belief] is wrong 🗿"
- "They found [X] in [place] — and then destroyed the evidence 🔍" """,
}

DESC_STYLES = {
    "A": "story_tease",
    "B": "fact_expand",
    "C": "news_breakdown",
    "D": "mystery_tease",
}

def generate_title_and_description(post, script, niche="A", mood="dramatic"):
    """Generate stunning titles, description, and hashtags. Returns dict."""
    niche = niche.upper()

    from niche_config import NICHES
    base_hashtags = NICHES[niche]["hashtags_base"]

    prompt = f"""You are a YouTube Shorts optimization expert. Generate viral titles and a description.

Niche: {NICHES[niche]['name']}
Script excerpt: {script[:300]}
Original topic: {post['title']}
Mood: {mood}

OUTPUT FORMAT (follow exactly, no extra text):

TITLE_1: [title here]
TITLE_2: [title here]
TITLE_3: [title here]

DESCRIPTION:
[3-4 punchy sentences:
 - Tease the story/fact without spoiling it
 - Add 1 line of context or stakes
 - End with a question to drive comments
 - Then on new line: {' '.join(base_hashtags[:5])}
]

HASHTAGS: [8-10 hashtags including {' '.join(base_hashtags[:3])} plus topic-specific ones]

PINNED_COMMENT: [1 engaging question to pin as first comment to boost engagement]

Title rules:
{TITLE_STYLES.get(niche, TITLE_STYLES['A'])}
- Under 60 characters each
- Add 1 emoji at the end
- Never mislead — must match the actual content
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.9
    )

    raw    = response.choices[0].message.content.strip()
    result = {
        "titles":         [],
        "description":    "",
        "hashtags":       "",
        "pinned_comment": "",
    }

    lines = raw.split("\n")
    mode  = None

    for line in lines:
        line = line.strip()
        if line.startswith("TITLE_"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                result["titles"].append(parts[1].strip())
        elif line.startswith("DESCRIPTION:"):
            mode = "desc"
        elif line.startswith("HASHTAGS:"):
            mode = None
            result["hashtags"] = line.replace("HASHTAGS:", "").strip()
        elif line.startswith("PINNED_COMMENT:"):
            mode = None
            result["pinned_comment"] = line.replace("PINNED_COMMENT:", "").strip()
        elif mode == "desc" and line:
            result["description"] += line + "\n"

    result["description"] = result["description"].strip()
    return result


def generate_hashtags(post, mood, niche="A"):
    """Quick hashtag generation (used in main.py for backwards compat)."""
    from niche_config import NICHES
    return " ".join(NICHES[niche.upper()]["hashtags_base"])
