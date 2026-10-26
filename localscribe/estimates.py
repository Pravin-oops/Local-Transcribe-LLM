"""Reading-time estimates for the selected outputs (shown before and after a run)."""
import re
from pathlib import Path

from localscribe.config import SPOKEN_WPM

READING_WPM = 220  # silent reading, for summary and study notes


def estimate_minutes(output: str, duration_s, story_minutes: int) -> float:
    """Rough reading time of an output before it exists. Length grows with the
    recording: about one chapter per 3 minutes and more concepts per hour."""
    hours = (duration_s or 3600) / 3600
    if output == "story":
        return float(story_minutes)
    if output == "summary":
        return (280 + 22 * 20 * hours) / READING_WPM   # overview + key points + ~20 chapters/hour
    if output == "notes":
        return (450 + 350 * hours) / READING_WPM       # concepts, takeaways, questions
    return 0.0


def actual_minutes(path: Path, output: str) -> float:
    """Reading time of a finished output file (the story is timed for reading aloud)."""
    text = re.sub(r"[#*_>\-\[\]()`|]", " ", Path(path).read_text(encoding="utf-8"))
    words = len(text.split())
    return words / (SPOKEN_WPM if output == "story" else READING_WPM)


def reading_summary(outputs, duration_s, story_minutes: int, job_dir: Path = None) -> str:
    """'About 22 min of reading · Engaging Story 15 · Summary 3 · Study notes 4'.

    Uses the real files when they exist for this recording, estimates otherwise.
    """
    from localscribe.config import OUTPUTS

    parts, total, estimated = [], 0.0, False
    for key in outputs:
        path = Path(job_dir) / f"{key}.md" if job_dir else None
        if path and path.exists():
            minutes = actual_minutes(path, key)
        else:
            minutes, estimated = estimate_minutes(key, duration_s, story_minutes), True
        total += minutes
        parts.append(f"{OUTPUTS[key]} {max(1, round(minutes))}")
    if not parts:
        return "Transcript only"
    lead = "About" if estimated else "Total"
    return f"{lead} {max(1, round(total))} min of reading · " + " · ".join(parts)
