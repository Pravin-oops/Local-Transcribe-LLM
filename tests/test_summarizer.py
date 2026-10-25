from localscribe.jobs import Job
from localscribe.summarizer import (clean_chapters, merge_chapters, render_chapters, render_notes,
                                    split_blocks, summarize)

from fakes import FakeBackend, fake_segments


def _count_words(text):
    return len(text.split())


def test_split_blocks_respects_budget_and_order():
    lines = [{"start": i, "end": i + 1, "text": "one two three four"} for i in range(10)]
    blocks = split_blocks(lines, _count_words, max_tokens=12)
    assert [l["start"] for b in blocks for l in b] == list(range(10))
    assert all(sum(_count_words(f"[x] {l['text']}") + 1 for l in b) <= 12 for b in blocks)


def test_clean_chapters_snaps_and_rejects_out_of_range_times():
    block = [{"start": 600.0, "end": 630.0, "text": "a"}, {"start": 631.0, "end": 660.0, "text": "b"}]
    raw = [{"start": "00:10:29", "title": "Near second line", "summary": ""},
           {"start": "02:00:00", "title": "Hallucinated time", "summary": ""},
           {"start": "nonsense", "title": "", "summary": ""}]
    out = clean_chapters(raw, block)
    assert [(c["start"], c["title"]) for c in out] == [(600.0, "Hallucinated time"),
                                                      (631.0, "Near second line")]


def test_chapter_start_refined_to_segment_mentioning_title():
    # Real case: the model can only pick the 80.76 s line, but the topic starts at 96.91 s.
    segments = [{"start": 80.76, "end": 85.16, "text": "The key idea here is that evaporation requires energy."},
                {"start": 90.59, "end": 95.87, "text": "Remember that term, latent heat."},
                {"start": 96.91, "end": 104.11, "text": "The second stage is condensation."},
                {"start": 111.0, "end": 115.0, "text": "Next line."}]
    block = [{"start": 80.76, "end": 104.11, "text": "..."}, {"start": 111.0, "end": 115.0, "text": "..."}]
    out = clean_chapters([{"start": "00:01:20", "title": "Condensation", "summary": ""}], block, segments)
    assert out[0]["start"] == 96.91
    # No title word in the line -> keep the line start.
    out = clean_chapters([{"start": "00:01:20", "title": "Big Picture", "summary": ""}], block, segments)
    assert out[0]["start"] == 80.76


def test_chapter_start_moves_back_to_first_distinctive_mention():
    # Real case: model picked 4:05 for "Why the Water Cycle Matters"; the topic starts at 3:45.
    filler = [{"start": float(t), "end": t + 4.0, "text": "Water moves through the water cycle."}
              for t in range(150, 225, 5)]
    segments = sorted(filler + [
        {"start": 225.5, "end": 229.0, "text": "So why does all of this matter?"},
        {"start": 230.0, "end": 236.0, "text": "First, it moves heat around the planet."},
        {"start": 245.0, "end": 250.0, "text": "Second, it delivers fresh water."},
    ], key=lambda s: s["start"])
    block = [{"start": 150.0, "end": 224.0, "text": "..."}, {"start": 245.0, "end": 260.0, "text": "..."}]
    chs = [{"start": "00:02:30", "title": "The Water Cycle Stages", "summary": ""},
           {"start": "00:04:05", "title": "Why the Water Cycle Matters", "summary": ""}]
    out = clean_chapters(chs, block, segments)
    # 'water'/'cycle' are everywhere, so only 'matter' counts and it is found 20 s earlier.
    assert [c["start"] for c in out] == [150.0, 225.5]


def test_merge_chapters_drops_repeats():
    chs = [{"start": 0, "title": "Intro", "summary": ""}, {"start": 50, "title": "intro", "summary": ""},
           {"start": 90, "title": "Methods", "summary": ""}]
    assert [c["title"] for c in merge_chapters(chs)] == ["Intro", "Methods"]


def test_summarize_caches_map_results(tmp_path):
    job = Job(tmp_path)
    backend = FakeBackend(n_ctx=3600)  # 1800-token blocks -> several blocks
    events = []
    result = summarize(fake_segments(400), backend, job, "Lecture", lambda f, m: events.append(f))
    assert backend.calls.count("map") >= 3
    assert "condense" not in backend.calls
    assert result["notes"]["overview"] == "Overview."
    assert result["chapters"] == sorted(result["chapters"], key=lambda c: c["start"])
    assert events[-1] == 1.0

    again = FakeBackend(n_ctx=3600)
    summarize(fake_segments(400), again, job, "Lecture", lambda f, m: None)
    assert "map" not in again.calls  # resumed from notes/*.json


def test_reduce_condenses_when_notes_do_not_fit(tmp_path):
    backend = FakeBackend(n_ctx=2620)  # leaves ~20 tokens for the notes digest
    summarize(fake_segments(200), backend, Job(tmp_path), "Lecture", lambda f, m: None)
    assert "condense" in backend.calls
    assert backend.calls[-1] == "notes"


def test_render_outputs():
    md = render_chapters([{"start": 75, "title": "Intro", "summary": "Hi."}])
    assert "- **[00:01:15] Intro** — Hi." in md
    notes = render_notes({"overview": "O", "key_concepts": [{"term": "T", "explanation": "E"}],
                          "key_takeaways": ["K"], "examples": [], "review_questions": ["Q?"]})
    assert "- **T**: E" in notes and "1. Q?" in notes and "### Examples" not in notes
    assert "could not be generated" in render_notes(None)
