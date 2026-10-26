"""Prompts and JSON schemas for the LLM outputs.

Structured steps are constrained to these schemas with llama.cpp grammars, so
the model cannot return malformed JSON or drift into free text. Map schemas are
built from only the fields the selected outputs need.
"""

# Bump when prompts or schemas change, so cached per-block results are regenerated.
VERSION = 8

SYSTEM = (
    "You are a careful note-taker who turns spoken English transcripts (lectures, talks, "
    "meetings, podcasts) into accurate notes. Use only information stated in the transcript. "
    "Do not invent facts, names, numbers or examples. Write in clear, plain English."
)

SPEAKER_NOTE = ("Lines may start with a speaker label such as 'Speaker 2:'; use names from the "
                "dialogue instead of labels when they are clear.")

_STRING_LIST = {"type": "array", "items": {"type": "string"}}
_CONCEPTS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}},
        "required": ["term", "explanation"],
    },
}
_CHAPTERS = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"start": {"type": "string"}, "title": {"type": "string"},
                       "summary": {"type": "string"}},
        "required": ["start", "title", "summary"],
    },
}


def _object(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


# ---------------------------------------------------------------- summary & study notes

FIELD_SCHEMAS = {"chapters": _CHAPTERS, "key_points": _STRING_LIST,
                 "concepts": _CONCEPTS, "examples": _STRING_LIST}
OUTPUT_FIELDS = {"summary": ("chapters", "key_points"),
                 "notes": ("key_points", "concepts", "examples")}


def fields_for(outputs) -> list:
    """Map-step fields needed by the selected outputs, in a stable order."""
    wanted = {f for o in outputs for f in OUTPUT_FIELDS.get(o, ())}
    return [f for f in FIELD_SCHEMAS if f in wanted]


def map_schema(fields) -> dict:
    return _object({f: FIELD_SCHEMAS[f] for f in fields})


def condense_schema(fields) -> dict:
    return _object({f: FIELD_SCHEMAS[f] for f in fields if f != "chapters"})


SUMMARY_SCHEMA = _object({"overview": {"type": "string"}, "key_points": _STRING_LIST})

NOTES_SCHEMA = _object({"key_concepts": _CONCEPTS, "key_takeaways": _STRING_LIST,
                        "examples": _STRING_LIST, "review_questions": _STRING_LIST})


def map_prompt(lines_text: str, start: str, end: str, part: int, total: int, minutes: float,
               fields) -> str:
    n_chapters = f"{max(2, round(minutes / 4))}-{max(4, round(minutes / 2))}"
    n_points = max(5, round(minutes * 1.5))
    instructions = {
        "chapters": f'- "chapters": start a new chapter each time the speaker moves on to a new topic or stage (usually {n_chapters} chapters here), in order from the first line to the last. Keep separate topics in separate chapters. "start" must be copied exactly from the timestamp of the line where that topic begins (format HH:MM:SS). "title" is 3-8 words naming the topic. "summary" is 1-2 sentences.',
        "key_points": f'- "key_points": about {n_points} important ideas covering the excerpt from start to finish, including its final part. Each is a short self-contained sentence.',
        "concepts": '- "concepts": every technical term or named idea that is defined or explained, each with a 1-2 sentence explanation based on what was said.',
        "examples": '- "examples": concrete examples, cases, demonstrations or numbers that were mentioned. Use [] if none.',
    }
    asked = "\n".join(instructions[f] for f in fields)
    return f"""Transcript excerpt {part} of {total}, covering {start} to {end} ({minutes:.0f} minutes).
Each line begins with its [HH:MM:SS] timestamp. {SPEAKER_NOTE}

<transcript>
{lines_text}
</transcript>

Return JSON with:
{asked}"""


def condense_prompt(notes_text: str, fields) -> str:
    names = ", ".join(f'"{f}"' for f in fields if f != "chapters")
    return f"""Below are notes from consecutive parts of one recording.

<notes>
{notes_text}
</notes>

Merge them into one shorter set of notes. Remove duplicates, combine related points, and keep every distinct important idea, term and example. Return JSON with {names}."""


def summary_prompt(notes_text: str, title: str) -> str:
    return f"""Below are notes taken, part by part, from the recording "{title}".

<notes>
{notes_text}
</notes>

Write the summary of the whole recording. Return JSON with:
- "overview": one or two paragraphs (4-8 sentences) explaining what the recording is about, how it develops and its main message.
- "key_points": the 6-12 most important points, in the order they come up.
Cover the whole recording, including its final parts."""


def notes_prompt(notes_text: str, title: str) -> str:
    return f"""Below are notes taken, part by part, from the recording "{title}".

<notes>
{notes_text}
</notes>

Write the final study notes for the whole recording. Return JSON with:
- "key_concepts": the most important terms and ideas, deduplicated, each with a clear 1-3 sentence explanation. Order them as they appear in the recording.
- "key_takeaways": 5-12 points a student should remember.
- "examples": the most useful concrete examples mentioned. Use [] if none.
- "review_questions": 5-10 questions that test understanding (why and how, not just recalling a number) and can be answered from these notes.
Cover the whole recording, including its final parts."""


# ---------------------------------------------------------------- Engaging Story

STORY_SYSTEM = (
    "You are a master storyteller who retells films as gripping narrated stories for an "
    "audience. Stay faithful to the events in the material you are given: do not invent major "
    "plot events, characters or endings. You may add atmosphere, sensory detail and emotion."
)

BEATS_SCHEMA = _object({
    "events": {"type": "array", "items": _object({
        "start": {"type": "string"}, "event": {"type": "string"}})},
    "scenes": {"type": "array", "items": _object({
        "start": {"type": "string"}, "setting": {"type": "string"}, "mood": {"type": "string"}})},
    "characters": {"type": "array", "items": _object({
        "name": {"type": "string"}, "speaker": {"type": "string"}, "role": {"type": "string"}})},
    "memorable_lines": _STRING_LIST,
})

OUTLINE_SCHEMA = _object({
    "title": {"type": "string"},
    "logline": {"type": "string"},
    "genre": {"type": "string"},
    "characters": {"type": "array", "items": _object({
        "name": {"type": "string"}, "description": {"type": "string"}})},
    "themes": _STRING_LIST,
})

TONE_STYLES = {
    "suspense": "tense and suspenseful: short sentences, slow reveals, cliffhangers",
    "epic": "sweeping and epic: grand imagery, rising stakes, a heroic rhythm",
    "drama": "emotional and intimate: feelings, relationships and hard choices",
    "comedy": "light-hearted and witty: playful asides and good comic timing",
    "dark": "dark and eerie: dread, shadows and unsettling detail",
    "bedtime": "gentle and warm, suitable for children: soft pacing, nothing graphic, a comforting close",
    "documentary": "clear and factual like a documentary narrator: who did what, and why",
}


def tone_style(tone: str, genre: str) -> str:
    return TONE_STYLES.get(tone) or f"captivating, matching the film's genre ({genre or 'drama'})"


def beats_prompt(lines_text: str, start: str, end: str, part: int, total: int, minutes: float) -> str:
    n_events = max(6, round(minutes * 3))
    return f"""Transcript excerpt {part} of {total} of a film's audio, covering {start} to {end} ({minutes:.0f} minutes).
Only the spoken dialogue was transcribed; nothing describes what is seen, so infer the action from what is said.
Each line begins with its [HH:MM:SS] timestamp. Lines may start with a speaker label such as 'Speaker 2:'.

<transcript>
{lines_text}
</transcript>

Return JSON with:
- "events": about {n_events} plot events in order, covering the excerpt from its first line to its last. Each "event" is one sentence saying who does, says, discovers or decides what, using character names (never speaker labels). Include every turning point, danger, secret and revelation. "start" is copied exactly from the timestamp of the line where it happens.
- "scenes": where the action takes place, one entry each time the place, time or situation changes: "start" (copied timestamp), "setting" (where or when, "unknown" if unclear) and "mood" (one or two words).
- "characters": each person once, with "name", "speaker" (the speaker label they talk under, if you can tell, else "") and "role" (who they are and what they want, one sentence). A bare label like "Speaker 3" is never a name: if a name is never said, name the person by role, for example "the ship's captain". One person may use several labels.
- "memorable_lines": up to 3 striking lines of dialogue, each under 15 words, copied exactly."""


def outline_prompt(events_text: str, characters_text: str, source_title: str) -> str:
    return f"""Below are the plot events of a whole film (source file "{source_title}"), in order, followed by the characters noticed along the way.

<events>
{events_text}
</events>

<characters>
{characters_text}
</characters>

Return JSON with:
- "title": an evocative title for the retelling (use the film's own title if the dialogue reveals it).
- "logline": one gripping sentence that sets up the story without revealing the ending.
- "genre": the film's genre as best you can tell (for example thriller, romance, drama, comedy, science fiction, horror, adventure, mystery).
- "characters": the main characters. Merge duplicates, and never list a bare speaker label like "Speaker 3" as a character. "description" is one sentence.
- "themes": 2-4 themes of the story."""


def section_prompt(*, part: int, total: int, title: str, genre: str, style: str, characters: str,
                   story_so_far: str, previous_tail: str, events: str, lines: str, words: int,
                   last_event: str) -> str:
    if part == 1:
        opening = "Open with a hook of one to three sentences that makes the listener need to know what happens next."
    else:
        opening = "Continue smoothly from where the previous part stopped; do not recap it or start with a new hook."
    if part == total:
        closing = (f'This is the final part: narrate every event through to the last one ("{last_event}"), '
                   "then bring the story to its ending and close with a memorable final line.")
    else:
        closing = (f'Narrate every event through to the last one ("{last_event}") and end this part right '
                   "after it, on a moment of tension or curiosity that pulls the listener into the next part.")
    return f"""You are writing part {part} of {total} of a narrated retelling of the film "{title}" (genre: {genre}).
Narration style: {style}.

Main characters:
{characters}

Story so far:
{story_so_far or "(this is the beginning)"}

The previous part ended with: "{previous_tail or "(nothing yet)"}"

Events to narrate in this part, in order:
{events}

Dialogue spoken during this part, in order:
{lines or "(none)"}

Write about {words} words of narrator script for this part. {opening} {closing}
Tell it as a third-person storyteller speaking to an audience, in the past tense, with vivid but clear language. Narrate every event in the list, in that order, using the dialogue to fill in what happens; add no new events, objects or people. Atmosphere, feelings and sensory detail are welcome. Retell the dialogue mostly in your own words: quote at most four short lines, copied word for word from the dialogue above and credited to the person who said them, and never put any other words in quotation marks. Labels like "Speaker 3" are placeholders, never names: do not mention them. Use the cue [pause] at most three times for dramatic effect; use no other stage directions.
Start with a heading line in the form "## <short evocative title for this part>", then the script. Output only the heading and the script."""
