"""Timestamp formatting and parsing."""
import re


def _split(seconds: float):
    total_ms = max(0, round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return h, m, s, ms


def hms(seconds: float) -> str:
    """12345.6 -> '03:25:45' (truncates to the whole second)."""
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    h, m, s, ms = _split(seconds)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


_HMS_RE = re.compile(r"^\s*\[?\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,]\d+)?\s*\]?\s*$")


def parse_hms(text: str):
    """'01:02:03', '2:03', '[00:12:30]' -> seconds; None if not a timestamp."""
    if not isinstance(text, str):
        return None
    m = _HMS_RE.match(text)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mins, secs = int(m.group(2)), int(m.group(3))
    if mins >= 60 or secs >= 60:
        return None
    return float(h * 3600 + mins * 60 + secs)
