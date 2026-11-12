"""YouTube links as a source: the video's captions if it has any, else its audio.

``fetch`` turns a link into a local file under ``<output_dir>/youtube`` that the
normal pipeline then processes:

- captions found (the creator's own are preferred over auto-generated ones):
  saved as ``<title> [<id>].vtt`` and used as the transcript, so nothing is
  transcribed and a summary is ready in minutes
- no captions: the smallest good audio track is downloaded as
  ``<title> [<id>].webm`` (or .m4a) and transcribed with Whisper like any file

The saved file is reused on the next run, so a job resumes without the network.
This is the only part of LocalScribe that goes online. yt-dlp is imported
lazily: it is only needed when a link is actually used.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from localscribe import subtitles
from localscribe.audio import AUDIO_EXTENSIONS

_URL = re.compile(r"^(?:https?://)?(?:www\.|m\.|music\.)?"
                  r"(?:youtube(?:-nocookie)?\.com/(?:watch\?(?:[^#\s]*&)?v=|shorts/|embed/|live/|v/)|youtu\.be/)"
                  r"([\w-]{11})(?![\w-])", re.IGNORECASE)
_NON_SPEECH = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # a whole cue like "[Music]" or "(applause)"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Speech survives low bitrates, and the connection may be slow: ~50 kbps Opus is
# about 22 MB per hour. yt-dlp prefers the original-language track among dubs.
AUDIO_FORMAT = "bestaudio[abr<=64]/bestaudio[abr<=96]/bestaudio/best"
CAPTION_KINDS = {"manual": "creator-provided", "auto": "auto-generated"}


class YouTubeError(RuntimeError):
    pass


@dataclass
class Fetched:
    path: Path
    kind: str                                   # "captions" or "audio"
    notes: list = field(default_factory=list)   # what happened, for the job log


def video_id(text):
    """The 11-character video id in a YouTube link, or None if it is not one."""
    m = _URL.match(str(text).strip())
    return m.group(1) if m else None


def is_url(text) -> bool:
    return video_id(text) is not None


def watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def downloads_dir(output_dir) -> Path:
    return Path(output_dir) / "youtube"


def cached(vid: str, dest_dir):
    """A file already fetched for this video (captions win over audio), or None."""
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        return None
    files = [p for p in sorted(dest_dir.iterdir()) if p.is_file() and p.stem.endswith(f"[{vid}]")]
    for extensions in ((".vtt",), AUDIO_EXTENSIONS):
        for p in files:
            if p.suffix.lower() in extensions:
                return p
    return None


# ---------------------------------------------------------------------- captions

def caption_tracks(info: dict, language: str = "en") -> list:
    """[(kind, lang, url)] of json3 caption tracks in ``language``, best first."""
    language = (language or info.get("language") or "en").split("-")[0].lower()

    def matching(table: dict, first: list) -> list:
        langs = [l for l in first if l in table]
        langs += sorted(l for l in table if l.lower().startswith(language + "-") and l not in langs)
        return [(l, t["url"]) for l in langs for t in table[l] if t.get("ext") == "json3" and t.get("url")]

    manual = matching(info.get("subtitles") or {}, [language])
    # "<lang>-orig" is what was actually said; plain "<lang>" may be a machine translation.
    auto = matching(info.get("automatic_captions") or {}, [f"{language}-orig", language])
    return [("manual", *t) for t in manual] + [("auto", *t) for t in auto]


def parse_json3(data: dict) -> list:
    """YouTube's json3 caption format -> [{"start", "end", "text"}].

    Auto-generated captions scroll: each line stays on screen while the next one
    appears, so a cue is cut where the following one starts.
    """
    raw = []
    for event in data.get("events") or []:
        text = "".join(s.get("utf8", "") for s in event.get("segs") or [])
        text = " ".join(text.replace(">>", " ").replace("♪", " ").replace("-->", "→").split())
        if not re.search(r"\w", text) or _NON_SPEECH.fullmatch(text):
            continue
        start = event.get("tStartMs", 0) / 1000
        raw.append((start, start + event.get("dDurationMs", 0) / 1000, text))
    raw.sort()
    cues = []
    for i, (start, end, text) in enumerate(raw):
        if i + 1 < len(raw):
            end = min(end, raw[i + 1][0])
        cues.append({"start": round(start, 2), "end": round(max(end, start + 0.1), 2), "text": text})
    return cues


def _stamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def to_vtt(cues: list, url: str, kind: str) -> str:
    head = f"WEBVTT\n\nNOTE\nsource: {url}\ncaptions: {CAPTION_KINDS.get(kind, kind)}\n"
    return head + "".join(f"\n{_stamp(c['start'])} --> {_stamp(c['end'])}\n{c['text']}\n" for c in cues)


def load_captions(path) -> list:
    """Transcript segments from a captions file saved by ``fetch``.

    Unlike movie subtitles there is no dialogue or sound-cue cleanup: a lecture's
    "f(x)" must stay in the text, and consecutive cues join into normal lines.
    """
    path = Path(path)
    segments = []
    for cue in subtitles.parse_cues(subtitles.read_text(path)):
        text = " ".join(subtitles._TAG.sub("", " ".join(cue["lines"])).split())
        if text:
            segments.append({"start": cue["start"], "end": cue["end"], "text": text})
    if not segments:
        raise subtitles.SubtitleError(f"No captions found in {path.name}")
    return segments


def caption_kind(path) -> str:
    """'creator-provided' / 'auto-generated' as recorded in a saved captions file."""
    try:
        m = re.search(r"^captions: (.+)$", subtitles.read_text(Path(path))[:500], re.MULTILINE)
    except OSError:
        return ""
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------- network

class _Quiet:
    """yt-dlp logger: failures are raised as exceptions, the rest is noise."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _session(**options):
    try:
        import yt_dlp
    except ImportError as exc:
        raise YouTubeError("YouTube links need the yt-dlp package: pip install yt-dlp") from exc
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noprogress": True, "logger": _Quiet(),
                             "noplaylist": True, "socket_timeout": 30, "retries": 5,
                             # No ffmpeg post-processing: PyAV decodes the download as it is.
                             "fixup": "never", **options})


def _reason(exc) -> str:
    text = _ANSI.sub("", str(exc)).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    return re.sub(r"^\[[\w:]+\]\s*[\w-]{11}:\s*", "", text) or type(exc).__name__


def _extract(ydl, url: str, download: bool = False) -> dict:
    from yt_dlp.utils import DownloadCancelled
    try:
        info = ydl.extract_info(url, download=download)
    except DownloadCancelled:
        raise
    except Exception as exc:
        raise YouTubeError(f"YouTube: {_reason(exc)}") from exc
    if not info or info.get("_type") == "playlist":
        raise YouTubeError("That link is not a single YouTube video")
    return info


def _file_title(title: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f\[\]%]+', " ", title)
    return " ".join(cleaned.split())[:80].strip(" .") or "video"


def probe(url: str, language: str = "en") -> dict:
    """Look a video up without downloading anything:
    {"id", "title", "duration", "captions": "manual" | "auto" | ""}."""
    vid = video_id(url)
    if not vid:
        raise YouTubeError("That does not look like a YouTube video link")
    with _session(skip_download=True) as ydl:
        info = _extract(ydl, watch_url(vid))
    tracks = caption_tracks(info, language)
    return {"id": vid, "title": info.get("title") or vid, "duration": info.get("duration"),
            "captions": tracks[0][0] if tracks else ""}


def fetch(url: str, dest_dir, language: str = "en", on_progress=None, cancel=None) -> Fetched:
    """Captions (or, without any, the audio) of a video as a local file in ``dest_dir``.

    ``on_progress(message)`` reports what is happening; setting ``cancel`` stops
    an audio download, whose partial file is continued by the next call.
    """
    vid = video_id(url)
    if not vid:
        raise YouTubeError(f"Not a YouTube video link: {url}")
    say = on_progress or (lambda message: None)
    dest_dir = Path(dest_dir)
    hit = cached(vid, dest_dir)
    if hit:
        kind = "captions" if hit.suffix.lower() == ".vtt" else "audio"
        return Fetched(hit, kind, [f"YouTube: using the {kind} saved earlier ({hit.name})"])

    dest_dir.mkdir(parents=True, exist_ok=True)
    url, notes = watch_url(vid), []
    say("Checking YouTube for captions")
    with _session(skip_download=True) as ydl:
        info = _extract(ydl, url)
        if info.get("is_live"):
            raise YouTubeError("This is a live stream: try again once it has ended")
        name = f"{_file_title(info.get('title') or vid)} [{vid}]"
        tracks = caption_tracks(info, language)
        for kind, lang, track_url in tracks:
            say(f"Downloading {CAPTION_KINDS[kind]} captions")
            try:
                cues = parse_json3(json.loads(ydl.urlopen(track_url).read()))
            except Exception as exc:  # blocked or empty track: try the next one
                notes.append(f"YouTube: {CAPTION_KINDS[kind]} captions ({lang}) could not be read: {_reason(exc)}")
                continue
            if cues:
                path = dest_dir / f"{name}.vtt"
                path.write_text(to_vtt(cues, url, kind), encoding="utf-8")
                notes.append(f"YouTube: {CAPTION_KINDS[kind]} captions found ({lang}, {len(cues)} lines)")
                return Fetched(path, "captions", notes)
            notes.append(f"YouTube: {CAPTION_KINDS[kind]} captions ({lang}) are empty")
    notes.append("YouTube: " + ("no usable captions" if tracks else "this video has no captions")
                 + "; downloading the audio to transcribe it instead")
    return Fetched(_download_audio(url, dest_dir, name, say, cancel), "audio", notes)


def _download_audio(url: str, dest_dir: Path, name: str, say, cancel) -> Path:
    from yt_dlp.utils import DownloadCancelled

    last = [-1]

    def hook(d: dict) -> None:
        if cancel is not None and cancel.is_set():
            raise DownloadCancelled()
        if d.get("status") != "downloading":
            return
        done, total = d.get("downloaded_bytes") or 0, d.get("total_bytes") or d.get("total_bytes_estimate")
        percent = int(100 * done / total) if total else done // 2**20
        if percent != last[0]:  # at most one message per percent (or per MB)
            last[0] = percent
            say(f"Downloading audio {percent}% ({done / 2**20:.1f} of {total / 2**20:.1f} MB)" if total
                else f"Downloading audio ({done / 2**20:.1f} MB)")

    say("No captions: downloading audio")
    with _session(format=AUDIO_FORMAT, outtmpl=str(dest_dir / f"{name}.%(ext)s"),
                  progress_hooks=[hook]) as ydl:
        info = _extract(ydl, url, download=True)
    files = [Path(d["filepath"]) for d in info.get("requested_downloads") or [] if d.get("filepath")]
    if not files or not files[0].is_file():
        raise YouTubeError("YouTube: the audio download did not produce a file")
    return files[0]
