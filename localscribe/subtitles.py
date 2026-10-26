"""Read .srt / .vtt subtitle files as a ready-made transcript (no audio, no Whisper).

Cleanup turns subtitle conventions into transcript segments:

- encodings: UTF-8 (with or without BOM), UTF-16, Windows-1252
- markup: <i>, <font ...>, {\\an8} and similar tags are removed
- dash dialogue: "- Hi.\\n- Hello." becomes two segments sharing the cue's time
- speaker names: "TOMAS: Hold it." sets speaker "Tomas" (ALL-CAPS names only,
  so "Note: ..." is not mistaken for a speaker)
- sound cues from hearing-impaired subtitles, "[thunder rumbling]" or
  "(door slams)", become separate segments marked {"sound": True}: context
  for the story, never dialogue
"""
import re
from pathlib import Path

from localscribe.timefmt import hms

SUBTITLE_EXTENSIONS = (".srt", ".vtt")

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_TIMING = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")
_CUE = re.compile(r"\[[^\]]+\]|\([^)]+\)")
_NAME = re.compile(r"^([A-Z][A-Z0-9 .'\-]{1,30}):\s*(.*)$")
_MUSIC = re.compile(r"[♪♫#]+")


class SubtitleError(ValueError):
    pass


def is_subtitle(path) -> bool:
    return Path(path).suffix.lower() in SUBTITLE_EXTENSIONS


def read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _seconds(h, m, s, ms) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_cues(text: str) -> list:
    """Raw cues [{"start", "end", "lines": [...]}] from SRT or WebVTT text."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        lines = [l.strip() for l in block.strip().split("\n")]
        timing = next((i for i, l in enumerate(lines) if _TIMING.search(l)), None)
        if timing is None:
            continue  # WEBVTT header, NOTE/STYLE blocks, stray text
        m = _TIMING.search(lines[timing])
        start, end = _seconds(*m.groups()[:4]), _seconds(*m.groups()[4:])
        body = [l for l in lines[timing + 1:] if l]
        if body and end > start:
            cues.append({"start": round(start, 2), "end": round(end, 2), "lines": body})
    cues.sort(key=lambda c: c["start"])
    return cues


def _split_cue(lines: list) -> list:
    """A cue's lines -> utterances, splitting dash dialogue and joining wrapped lines."""
    text = "\n".join(_TAG.sub("", l) for l in lines)
    if re.search(r"(^|\n)\s*-\s*\S", text):
        parts = [p.strip() for p in re.split(r"(?:^|\n)\s*-\s*", text) if p.strip()]
    else:
        parts = [text]
    return [" ".join(p.split()) for p in parts]


def _utterance(text: str):
    """(speaker or None, sound cues, spoken text) for one utterance."""
    text = _MUSIC.sub(" ", text).strip()
    m = _NAME.match(text)
    speaker = None
    if m and not _CUE.fullmatch(m.group(1)):
        speaker, text = m.group(1).strip().title(), m.group(2)
    sounds = [c.strip() for c in _CUE.findall(text)]
    spoken = " ".join(_CUE.sub(" ", text).split()).strip(" -")
    return speaker, sounds, spoken


def to_segments(cues: list) -> list:
    """Cues -> transcript segments, spreading a multi-utterance cue's time by text length."""
    segments = []
    for cue in cues:
        pieces = []
        for utterance in _split_cue(cue["lines"]):
            speaker, sounds, spoken = _utterance(utterance)
            for s in sounds:
                inner = s[1:-1].strip()
                if inner and not _NAME.match(s[1:-1] + ":"):  # skip "[TOMAS]"-style name tags
                    pieces.append({"text": f"[{inner.lower()}]", "sound": True})
                elif inner and speaker is None:
                    speaker = inner.title()
            if re.search(r"\w", spoken):
                pieces.append({"text": spoken, **({"speaker": speaker} if speaker else {})})
        if not pieces:
            continue
        span, total = cue["end"] - cue["start"], sum(len(p["text"]) for p in pieces)
        t = cue["start"]
        for p in pieces:
            length = span * len(p["text"]) / total
            # "cue": each subtitle utterance is its own turn, so unnamed lines from
            # different speakers are never merged into one transcript line.
            segments.append({"start": round(t, 2), "end": round(t + length, 2), **p, "cue": True})
            t += length
    return segments


def load(path) -> list:
    """Parse a subtitle file into transcript segments, or raise SubtitleError."""
    path = Path(path)
    cues = parse_cues(read_text(path))
    if not cues:
        raise SubtitleError(f"No subtitle cues found in {path.name}")
    segments = to_segments(cues)
    if not any(not s.get("sound") for s in segments):
        raise SubtitleError(f"{path.name} contains no spoken lines")
    return segments


def describe(path) -> str:
    """Short summary for the UI, e.g. '412 lines · 1:58:03 · 3 named speakers'."""
    try:
        segments = load(path)
    except (OSError, SubtitleError) as exc:
        return f"Unreadable subtitles: {exc}"
    spoken = [s for s in segments if not s.get("sound")]
    names = {s["speaker"] for s in spoken if s.get("speaker")}
    parts = [f"{len(spoken)} line{'s' if len(spoken) != 1 else ''}", hms(segments[-1]["end"])]
    if names:
        parts.append(f"{len(names)} named speaker{'s' if len(names) > 1 else ''}")
    if len(spoken) < len(segments):
        cues = len(segments) - len(spoken)
        parts.append(f"{cues} sound cue{'s' if cues > 1 else ''}")
    return " · ".join(parts)
