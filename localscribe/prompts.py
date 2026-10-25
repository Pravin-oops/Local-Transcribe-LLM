"""Prompts and JSON schemas for map-reduce summarization.

Outputs are constrained to these schemas with llama.cpp grammars, so the
model cannot return malformed JSON or drift into free text.
"""

# Bump when prompts or schemas change, so cached per-block notes are regenerated.
VERSION = 3

SYSTEM = (
    "You are a careful note-taker who turns spoken English transcripts (lectures, talks, "
    "meetings) into accurate study material. Use only information stated in the transcript. "
    "Do not invent facts, names, numbers or examples. Write in clear, plain English."
)

_STRING_LIST = {"type": "array", "items": {"type": "string"}}
_CONCEPTS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}},
        "required": ["term", "explanation"],
    },
}

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["start", "title", "summary"],
            },
        },
        "key_points": _STRING_LIST,
        "concepts": _CONCEPTS,
        "examples": _STRING_LIST,
    },
    "required": ["chapters", "key_points", "concepts", "examples"],
}

CONDENSE_SCHEMA = {
    "type": "object",
    "properties": {"key_points": _STRING_LIST, "concepts": _CONCEPTS, "examples": _STRING_LIST},
    "required": ["key_points", "concepts", "examples"],
}

NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "key_concepts": _CONCEPTS,
        "key_takeaways": _STRING_LIST,
        "examples": _STRING_LIST,
        "review_questions": _STRING_LIST,
    },
    "required": ["overview", "key_concepts", "key_takeaways", "examples", "review_questions"],
}


def map_prompt(lines_text: str, start: str, end: str, part: int, total: int, minutes: float) -> str:
    n_chapters = f"{max(2, round(minutes / 4))}-{max(4, round(minutes / 2))}"
    n_points = max(5, round(minutes * 1.5))
    return f"""Transcript excerpt {part} of {total}, covering {start} to {end} ({minutes:.0f} minutes).
Each line begins with its [HH:MM:SS] timestamp.

<transcript>
{lines_text}
</transcript>

Return JSON with:
- "chapters": start a new chapter each time the speaker moves on to a new topic or stage (usually {n_chapters} chapters here), in order from the first line to the last. Keep separate topics in separate chapters. "start" must be copied exactly from the timestamp of the line where that topic begins (format HH:MM:SS). "title" is 3-8 words naming the topic. "summary" is 1-2 sentences.
- "key_points": about {n_points} important ideas covering the excerpt from start to finish, including its final part. Each is a short self-contained sentence.
- "concepts": every technical term or named idea that is defined or explained, each with a 1-2 sentence explanation based on what was said.
- "examples": concrete examples, cases, demonstrations or numbers that were mentioned. Use [] if none."""


def condense_prompt(notes_text: str) -> str:
    return f"""Below are study notes from consecutive parts of one recording.

<notes>
{notes_text}
</notes>

Merge them into one shorter set of notes. Remove duplicates, combine related points, and keep every distinct important idea, term and example. Return JSON with "key_points", "concepts" and "examples"."""


def notes_prompt(notes_text: str, title: str) -> str:
    return f"""Below are notes taken, part by part, from the recording "{title}".

<notes>
{notes_text}
</notes>

Write the final study notes for the whole recording. Return JSON with:
- "overview": 3-6 sentences describing what the recording covers and its main message.
- "key_concepts": the most important terms and ideas, deduplicated, each with a clear 1-3 sentence explanation. Order them as they appear in the recording.
- "key_takeaways": 5-12 points a student should remember.
- "examples": the most useful concrete examples mentioned. Use [] if none.
- "review_questions": 5-10 questions that test understanding (why and how, not just recalling a number) and can be answered from these notes.
Cover the whole recording, including its final parts."""
