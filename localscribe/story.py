"""Engaging Story: retell a whole recording (typically a film) as a narrator script.

1. Beats    per transcript block: fine-grained timestamped plot events, scene
            settings and moods, the characters (with the speaker labels they
            talk under) and a few memorable lines -> story/beats_NNNN.json
2. Outline  for the whole film: title, logline, inferred genre, main cast and
            themes -> story/outline.json
3. Script   written part by part over consecutive events. Each part sees the
            outline, a running "story so far", the end of the previous part and
            only the quotes said during its own stretch of time, so the script
            stays in order and any length fits an 8k context -> story/part_NN.json

Every step is cached, so a cancelled story resumes where it stopped.
"""
import re

from localscribe import prompts
from localscribe.config import SPOKEN_WPM, STORY_TONES
from localscribe.jobs import read_json, write_json_atomic
from localscribe.summarizer import (PROMPT_OVERHEAD_TOKENS, block_key, complete_structured, digest,
                                    format_line, transcript_blocks)
from localscribe.timefmt import hms, parse_hms

BEATS_OUTPUT_TOKENS = 1800  # ~36 one-sentence events for a full ~12 min block
OUTLINE_OUTPUT_TOKENS = 900
# Longest part one call writes (~1,700 tokens) while the prompt still fits 8k.
# With fewer events than parts the story comes out shorter rather than padded.
MAX_PART_WORDS = 900


def story_plan(minutes: int):
    """(total words, number of parts) for a story read aloud in ``minutes``:
    3 parts up to ~10 min, then one part per ~550 words, at most 10."""
    words = minutes * SPOKEN_WPM
    return words, max(3, min(10, round(words / 550)))
_LABEL = re.compile(r"\bSpeaker \d+\b")


def _block_time(text, block: list) -> float:
    """Model-written timestamp, validated against the block and snapped to a real line."""
    lo, hi = block[0]["start"], block[-1]["end"]
    t = parse_hms(str(text or ""))
    if t is None or not (lo - 1 <= t <= hi + 1):
        t = lo
    return min((l["start"] for l in block), key=lambda s: abs(s - t))


def _norm_words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def quote_time(quote: str, segments: list):
    """Start time of the transcript segment a quote was copied from, or None if not found."""
    words = set(_norm_words(quote))
    if not words:
        return None
    best, score = None, 0.0
    spoken = [s for s in segments if not s.get("sound")]
    for i, seg in enumerate(spoken):
        window = set(_norm_words(" ".join(s["text"] for s in spoken[i:i + 2])))
        overlap = len(words & window) / len(words)
        if overlap > score:
            best, score = seg["start"], overlap
    return best if score >= 0.6 else None


def split_even(items: list, n: int) -> list:
    """Split into at most ``n`` contiguous, non-empty groups of near-equal size."""
    n = max(1, min(n, len(items)))
    size, extra = divmod(len(items), n)
    groups, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        groups.append(items[start:end])
        start = end
    return groups


def split_heading(text: str, index: int):
    """Separate the '## Title' line the model was asked to start with from the script."""
    lines = text.strip().splitlines()
    for i, line in enumerate(lines[:4]):
        if line.lstrip().startswith("#"):
            heading = line.strip().lstrip("#").strip().strip("*\"'").strip()
            body = "\n".join(lines[i + 1:]).strip()
            return heading or f"Part {index + 1}", body
    return f"Part {index + 1}", text.strip()


def resolve_labels(text: str, names: dict) -> str:
    """Replace speaker labels the model linked to a name ("Speaker 1" -> "Elena")."""
    return _LABEL.sub(lambda m: names.get(m.group(0), m.group(0)), text)


def _event_lines(events: list, scenes: list) -> str:
    """'- event' lines, noting the setting and mood whenever they change. No
    timestamps: the writer copies them into the prose."""
    out, current = [], None
    for e in events:
        scene = next((s for s in reversed(scenes) if s["start"] <= e["start"]), None)
        place = f"({scene['setting']}; {scene['mood']}) " if scene and scene is not current else ""
        current = scene or current
        out.append(f"- {place}{e['event']}")
    return "\n".join(out)


_NAME = r"[A-Z][\w'-]*(?: [A-Z][\w'-]*){0,2}"  # one to three capitalised words
_SCRIPT_LINE = re.compile(
    rf'^\s*(?:(?P<lead>Speaker \d+|{_NAME})\s*(?:\((?P<paren>{_NAME})\))?\s*:\s*)?'
    rf'"(?P<quote>[^"]+)"\s*(?:\((?P<who>{_NAME})\))?\s*$')
_INLINE_CREDIT = re.compile(rf'(?<=")\s*\({_NAME}\)')  # '"Wait." (Elena Vasquez)' -> '"Wait."'
_TIMESTAMP = re.compile(r"\[\d{1,2}:\d\d(?::\d\d)?\]\s*")


def clean_script(text: str) -> str:
    """Turn formatting the model copied from its prompt back into narration:
    stray [HH:MM:SS] stamps, and screenplay lines such as
    'Speaker 1: "Keep it steady." (Tomas)' -> '"Keep it steady," Tomas said.'"""
    out = []
    for line in _TIMESTAMP.sub("", text).split("\n"):
        m = _SCRIPT_LINE.match(line)
        if m and (m.group("lead") or m.group("who")):
            names = [n.strip() for n in (m.group("who"), m.group("paren"), m.group("lead"))
                     if n and n.strip() and not _LABEL.fullmatch(n.strip())]
            quote = m.group("quote").strip()
            if names:
                if quote[-1] in ".":
                    quote = quote[:-1] + ","
                line = f'"{quote}" {names[0]} said.'
            else:
                line = f'"{quote}"'
        line = _INLINE_CREDIT.sub("", re.sub(r"\bSpeaker \d+:\s*", "", line))
        line = re.sub(r"(?i)^(\s*)someone said:\s*", r"\1", line)  # copied "Someone said:" prefix
        out.append(_LABEL.sub("someone", line))
    return "\n".join(out).strip()


def dialogue_lines(segments: list, names: dict, lo: float, hi: float, keep: set,
                   limit: int = 20) -> str:
    """The dialogue spoken in [lo, hi), credited to resolved names, thinned evenly
    to ``limit`` lines (always keeping the memorable lines whose times are in ``keep``)."""
    span = [s for s in segments
            if lo <= s["start"] < hi and not s.get("sound") and len(s["text"].split()) >= 3]
    if len(span) > limit:
        must = [s for s in span if s["start"] in keep]
        rest = [s for s in span if s["start"] not in keep]
        step = len(rest) / max(1, limit - len(must))
        span = sorted(must + [rest[int(i * step)] for i in range(limit - len(must))],
                      key=lambda s: s["start"])
    def line(s):
        label = s.get("speaker") or ""
        name = names.get(label) if _LABEL.fullmatch(label) else label  # subtitle names are real
        # Written as prose ('Tomas said: "..."') because the writer imitates the format;
        # an unknown speaker is left for the writer to infer from context.
        return f"- {name} said: \"{s['text']}\"" if name else f"- \"{s['text']}\""

    return "\n".join(line(s) for s in span)


def drop_repeats(text: str, previous: str) -> str:
    """Remove sentences copied from the previous part (small models often restate its ending)."""
    def norm(s):
        return " ".join(_norm_words(s))
    seen = {norm(s) for s in re.split(r"(?<=[.!?])\s+", previous) if len(s.split()) >= 5}
    kept = []
    for para in text.split("\n"):
        sentences = re.split(r"(?<=[.!?])\s+", para)
        para = " ".join(s for s in sentences if norm(s) not in seen)
        if para.strip() or not kept or kept[-1].strip():
            kept.append(para)
    return "\n".join(kept).strip()


def _fit_events(events: list, count_tokens, budget: int) -> str:
    """Event list for the outline prompt, merged pairwise until it fits."""
    lines = [f"- [{hms(e['start'])}] {e['event']}" for e in events]
    while count_tokens("\n".join(lines)) > budget and len(lines) > 1:
        lines = [" ".join(pair) for pair in zip(lines[::2], lines[1::2] + [""])]
    return "\n".join(lines)


def write_story(segments: list, backend, job, title: str, tone: str, minutes: int, on_progress,
                cancel=None, warn=print):
    """Build the Engaging Story for ``segments``, sized to be read aloud in about
    ``minutes``; returns a dict for :func:`render_story`."""
    blocks = transcript_blocks(segments, backend, BEATS_OUTPUT_TOKENS)
    if not blocks:
        return None
    total_words, n_parts = story_plan(minutes)
    steps = len(blocks) + 1 + n_parts
    story_dir = job.output("story")

    # 1. Beats per block
    events, scenes, cast, names, quotes = [], [], {}, {}, []
    for i, block in enumerate(blocks):
        path = story_dir / f"beats_{i:04d}.json"
        key = block_key(backend, block)
        cached = read_json(path)
        if cached and cached.get("key") == key:
            data = cached["raw"]
        else:
            message = f"Following the story, part {i + 1}/{len(blocks)}"
            on_progress(i / steps, message)
            block_minutes = (block[-1]["end"] - block[0]["start"]) / 60
            user = prompts.beats_prompt("\n".join(format_line(l) for l in block), hms(block[0]["start"]),
                                        hms(block[-1]["end"]), i + 1, len(blocks), block_minutes)
            data = complete_structured(backend, user, prompts.BEATS_SCHEMA, BEATS_OUTPUT_TOKENS, cancel,
                                       lambda n: on_progress(i / steps, f"{message} ({n} tokens)"),
                                       warn, system=prompts.STORY_SYSTEM) or {}
            write_json_atomic(path, {"key": key, "raw": data})
        for ch in data.get("characters", []):
            name, label = str(ch.get("name", "")).strip(), str(ch.get("speaker", "")).strip()
            if not name or _LABEL.fullmatch(name):
                continue  # a bare speaker label is not a character
            if _LABEL.fullmatch(label):
                names.setdefault(label, name)
            cast.setdefault(name.lower(), {"name": name, "description": str(ch.get("role", "")).strip()})
        events += [{"start": _block_time(e.get("start"), block), "event": str(e["event"]).strip()}
                   for e in data.get("events", []) if str(e.get("event", "")).strip()]
        scenes += [{"start": _block_time(s.get("start"), block),
                    "setting": str(s.get("setting", "") or "unknown").strip(),
                    "mood": str(s.get("mood", "")).strip()} for s in data.get("scenes", [])]
        for q in data.get("memorable_lines", [])[:3]:
            t = quote_time(str(q), segments)
            if t is not None:  # a "quote" not found in the transcript is not used
                quotes.append((t, str(q).strip()))
    for e in events:
        e["event"] = resolve_labels(e["event"], names)
    events.sort(key=lambda e: e["start"])
    scenes.sort(key=lambda s: s["start"])
    if not events:
        warn("No story events could be extracted, so the Engaging Story was skipped.")
        return None

    # 2. Outline
    on_progress(len(blocks) / steps, "Outlining the story")
    outline_path = story_dir / "outline.json"
    outline_key = {"model": backend.name, "prompts": prompts.VERSION,
                   "events": digest("\n".join(f"{e['start']} {e['event']}" for e in events))}
    cached = read_json(outline_path)
    if cached and cached.get("key") == outline_key:
        outline = cached["raw"]
    else:
        cast_text = "\n".join(f"- {c['name']}: {c['description']}" for c in cast.values()) or "(none noted)"
        budget = backend.n_ctx - OUTLINE_OUTPUT_TOKENS - PROMPT_OVERHEAD_TOKENS - backend.count_tokens(cast_text)
        outline = complete_structured(
            backend, prompts.outline_prompt(_fit_events(events, backend.count_tokens, budget), cast_text, title),
            prompts.OUTLINE_SCHEMA, OUTLINE_OUTPUT_TOKENS, cancel,
            lambda n: on_progress(len(blocks) / steps, f"Outlining the story ({n} tokens)"), warn,
            system=prompts.STORY_SYSTEM) or {}
        write_json_atomic(outline_path, {"key": outline_key, "raw": outline})
    story_title = str(outline.get("title") or title).strip()
    genre = str(outline.get("genre", "")).strip()
    characters = [c for c in (outline.get("characters") or list(cast.values()))
                  if c.get("name") and not _LABEL.fullmatch(str(c["name"]).strip())]
    cast_lines = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in characters[:12])
    style = prompts.tone_style(tone, genre)

    # 3. Script, part by part over consecutive events
    groups = split_even(events, n_parts)
    words = max(150, min(MAX_PART_WORDS, total_words // len(groups)))
    quote_times = {t for t, _ in quotes}
    parts, previous_tail, previous_body = [], "", ""
    for k, group in enumerate(groups):
        path = story_dir / f"part_{k:02d}.json"
        key = {"model": backend.name, "prompts": prompts.VERSION, "tone": tone, "minutes": minutes,
               "parts": len(groups), "title": story_title, "previous": digest(previous_tail),
               "events": digest("\n".join(f"{e['start']} {e['event']}" for e in group))}
        cached = read_json(path)
        if cached and cached.get("key") == key:
            # Older caches stored the cleaned body and heading separately.
            text = cached.get("raw") or f"## {cached.get('heading', '')}\n{cached.get('text', '')}"
        else:
            message = f"Writing the story, part {k + 1}/{len(groups)}"
            done = len(blocks) + 1 + k
            on_progress(done / steps, message)
            so_far = "\n".join(f"- {p['heading']}: " + " ".join(e["event"] for e in g)[:500]
                               for p, g in zip(parts, groups))
            lo = group[0]["start"] if k else float("-inf")
            hi = groups[k + 1][0]["start"] if k + 1 < len(groups) else float("inf")
            user = prompts.section_prompt(
                part=k + 1, total=len(groups), title=story_title, genre=genre or "drama", style=style,
                characters=cast_lines or "(unknown)", story_so_far=so_far, previous_tail=previous_tail,
                events=_event_lines(group, scenes), last_event=group[-1]["event"],
                lines=dialogue_lines(segments, names, lo, hi, quote_times), words=words)
            text = backend.complete_text(prompts.STORY_SYSTEM, user, max_tokens=int(words * 1.7) + 150,
                                         cancel=cancel,
                                         on_token=lambda n: on_progress(done / steps, f"{message} ({n} tokens)"))
            # Raw text is cached; cleanup runs on every assembly so later fixes apply too.
            write_json_atomic(path, {"key": key, "raw": text})
        heading, body = split_heading(text, k)
        body = drop_repeats(clean_script(body), previous_body)
        parts.append({"heading": heading, "text": body})
        previous_tail = " ".join(body.split()[-60:])
        previous_body = body
    on_progress(1.0, "Story complete")
    return {"title": story_title, "logline": str(outline.get("logline", "")).strip(), "genre": genre,
            "tone": STORY_TONES.get(tone, tone), "characters": characters, "parts": parts,
            "words": sum(len(p["text"].split()) for p in parts)}


def render_story(story: dict, source_line: str) -> str:
    if not story:
        return "# Engaging Story\n\n*The story could not be generated.*\n"
    minutes = max(1, round(story["words"] / SPOKEN_WPM))
    tone = story["tone"] if story["tone"] != STORY_TONES["auto"] else "matched to the film"
    out = [f"# {story['title']}", ""]
    if story["logline"]:
        out += [f"*{story['logline']}*", ""]
    out += [f"*Genre: {story['genre'] or 'unknown'} · Narration: {tone} · "
            f"{story['words']:,} words (about {minutes} min read aloud) · {source_line}*", ""]
    if story["characters"]:
        out += ["**Cast**", ""]
        out += [f"- **{c.get('name', '').strip()}** — {c.get('description', '').strip()}"
                for c in story["characters"] if c.get("name")]
        out.append("")
    out += ["---", ""]
    for part in story["parts"]:
        out += [f"## {part['heading']}", "", part["text"], ""]
    return "\n".join(out)
