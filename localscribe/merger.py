"""Combine per-chunk segments and write transcript files."""
from pathlib import Path

from localscribe.jobs import write_json_atomic
from localscribe.timefmt import hms, srt_time, vtt_time


def group_segments(segments: list, max_span: float = 30.0, max_gap: float = 3.0) -> list:
    """Join consecutive segments into lines of up to ``max_span`` seconds.

    A pause longer than ``max_gap``, a change of speaker, or a subtitle cue
    (each cue is a separate turn) always starts a new line. Returns dicts with
    start, end, text and (if labelled) speaker.
    """
    lines = []
    for seg in segments:
        if (lines and not seg.get("cue")
                and seg["start"] - lines[-1]["end"] <= max_gap
                and seg["end"] - lines[-1]["start"] <= max_span
                and seg.get("speaker") == lines[-1].get("speaker")):
            lines[-1]["end"] = seg["end"]
            lines[-1]["text"] += " " + seg["text"]
        else:
            lines.append(dict(seg))
    return lines


def labelled(seg: dict) -> str:
    """Segment text with its speaker label, if any ("Speaker 2: ...")."""
    return f"{seg['speaker']}: {seg['text']}" if seg.get("speaker") else seg["text"]


def to_txt(segments: list) -> str:
    paragraphs = group_segments(segments, max_span=60.0, max_gap=2.5)
    return "\n\n".join(f"[{hms(p['start'])}] {labelled(p)}" for p in paragraphs) + "\n"


def to_srt(segments: list) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{labelled(seg)}\n")
    return "\n".join(blocks)


def to_vtt(segments: list) -> str:
    cues = [f"{vtt_time(s['start'])} --> {vtt_time(s['end'])}\n{labelled(s)}\n" for s in segments]
    return "WEBVTT\n\n" + "\n".join(cues)


def write_transcripts(segments: list, job_root: Path) -> dict:
    files = {
        "transcript.txt": to_txt(segments),
        "transcript.srt": to_srt(segments),
        "transcript.vtt": to_vtt(segments),
    }
    paths = {}
    for name, text in files.items():
        path = Path(job_root) / name
        path.write_text(text, encoding="utf-8")
        paths[name] = str(path)
    json_path = Path(job_root) / "transcript.json"
    write_json_atomic(json_path, segments)
    paths["transcript.json"] = str(json_path)
    return paths
