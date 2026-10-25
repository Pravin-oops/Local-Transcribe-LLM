"""Stand-ins for Whisper and the LLM so pipeline logic tests run in seconds."""
import json

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


class FakeBackend:
    name = "fake-llm"

    def __init__(self, n_ctx: int = 8192):
        self.n_ctx = n_ctx
        self.calls = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def complete_json(self, system, user, schema, max_tokens, cancel=None, on_token=None):
        props = schema["properties"]
        if "chapters" in props:
            self.calls.append("map")
            first = user.split("<transcript>\n", 1)[1].split("]", 1)[0].lstrip("[")
            return json.dumps({
                "chapters": [{"start": first, "title": f"Topic at {first}", "summary": "S."},
                             {"start": "99:00:00", "title": "Bad time", "summary": ""}],
                "key_points": [f"Point from {first}"],
                "concepts": [{"term": "entropy", "explanation": "disorder"}],
                "examples": [],
            })
        if "overview" in props:
            self.calls.append("notes")
            return json.dumps({"overview": "Overview.", "key_concepts": [],
                               "key_takeaways": ["Remember this."], "examples": [],
                               "review_questions": ["Why?"]})
        self.calls.append("condense")
        return json.dumps({"key_points": ["Condensed."], "concepts": [], "examples": []})

    def close(self):
        pass


def fake_segments(n: int, every: float = 20.0) -> list:
    return [{"start": i * every, "end": i * every + every - 1,
             "text": f"Sentence number {i} about {hms(i * every)} with some extra words here."}
            for i in range(n)]
