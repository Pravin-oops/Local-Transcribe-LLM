"""Stand-ins for Whisper, the diarizer and the LLM so pipeline logic tests run in seconds."""
import json
import re

from localscribe.timefmt import hms


class FakeTranscriber:
    device, compute_type = "cpu", "int8"

    def __init__(self, calls: list, cancel_after: int = None, cancel=None):
        self.calls, self.cancel_after, self.cancel = calls, cancel_after, cancel

    def transcribe(self, audio, offset, language, beam_size, prompt=None, on_segment=None, cancel=None):
        self.calls.append(offset)
        seg = {"start": round(offset, 2), "end": round(offset + len(audio) / 16000, 2),
               "text": f"Talk about topic {len(self.calls)}."}
        if on_segment:
            on_segment(seg)
        if self.cancel_after and len(self.calls) >= self.cancel_after:
            self.cancel.set()
        return [seg]

    def close(self):
        pass


def fake_diarizer(calls: list):
    """Alternates two speakers every 30 s."""
    def run(samples, settings, on_progress, cancel, on_status):
        calls.append(len(samples))
        on_progress(1.0)
        length = len(samples) / 16000
        return [{"start": float(t), "end": float(min(t + 30, length)), "speaker": (t // 30) % 2}
                for t in range(0, int(length), 30)]
    return run


class FakeBackend:
    name = "fake-llm"

    def __init__(self, n_ctx: int = 8192):
        self.n_ctx = n_ctx
        self.calls, self.schemas = [], []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete_json(self, system, user, schema, max_tokens, cancel=None, on_token=None):
        props = schema["properties"]
        self.schemas.append(sorted(props))
        stamps = re.findall(r"^\[(\d\d:\d\d:\d\d)\]", user, re.M)
        first = stamps[0] if stamps else "00:00:00"
        if "events" in props:
            self.calls.append("beats")
            picks = dict.fromkeys([first] + stamps[len(stamps) // 2: len(stamps) // 2 + 1] + stamps[-1:])
            return json.dumps({
                "events": [{"start": t, "event": f"Speaker 1 acts at {t}."} for t in picks],
                "scenes": [{"start": first, "setting": "a ship", "mood": "tense"}],
                "characters": [{"name": "Mara", "speaker": "Speaker 1", "role": "the captain"},
                               {"name": "Speaker 2", "speaker": "Speaker 2", "role": "a voice"}],
                "memorable_lines": ["Talk about topic 1.", "A line nobody said."]})
        if "logline" in props:
            self.calls.append("outline")
            return json.dumps({"title": "The Long Night", "logline": "A captain must choose.",
                               "genre": "thriller", "themes": ["duty"],
                               "characters": [{"name": "Mara", "description": "The captain."}]})
        if "overview" in props:
            self.calls.append("summary")
            return json.dumps({"overview": "Overview.", "key_points": ["Main point."]})
        if "review_questions" in props:
            self.calls.append("notes")
            return json.dumps({"key_concepts": [], "key_takeaways": ["Remember this."], "examples": [],
                               "review_questions": ["Why?"]})
        if "<transcript>" in user:
            self.calls.append("map")
            data = {"chapters": [{"start": first, "title": f"Topic at {first}", "summary": "S."},
                                 {"start": "99:00:00", "title": "Bad time", "summary": ""}],
                    "key_points": [f"Point from {first}"],
                    "concepts": [{"term": "entropy", "explanation": "disorder"}],
                    "examples": []}
            return json.dumps({k: v for k, v in data.items() if k in props})
        self.calls.append("condense")
        return json.dumps({k: (["Condensed."] if k == "key_points" else []) for k in props})

    def complete_text(self, system, user, max_tokens, cancel=None, on_token=None, temperature=0.8):
        self.calls.append("part")
        self.part_words = getattr(self, "part_words", []) + [int(re.search(r"about (\d+) words", user).group(1))]
        part = re.search(r"part (\d+) of (\d+)", user).group(1)
        return f"Here you go.\n## Chapter {part} Title\nThe night was long. " + "word " * 40

    def close(self):
        pass


def fake_segments(n: int, every: float = 20.0) -> list:
    return [{"start": i * every, "end": i * every + every - 1,
             "text": f"Sentence number {i} about {hms(i * every)} with some extra words here."}
            for i in range(n)]
