from localscribe.jobs import Job
from localscribe.story import (clean_script, dialogue_lines, drop_repeats, render_story, split_even,
                               split_heading, write_story)

from fakes import FakeBackend, fake_segments


def test_split_even_keeps_order_and_balance():
    groups = split_even(list(range(11)), 4)
    assert [len(g) for g in groups] == [3, 3, 3, 2]
    assert [x for g in groups for x in g] == list(range(11))
    assert split_even([1, 2], 5) == [[1], [2]]  # never more groups than items


def test_split_heading_drops_preamble():
    assert split_heading("Sure! Here it is.\n## **The Storm**\nRain fell.", 0) == ("The Storm", "Rain fell.")
    assert split_heading("No heading at all.", 2) == ("Part 3", "No heading at all.")


def test_dialogue_lines_credit_resolved_names_and_thin_evenly():
    segs = [{"start": float(i), "end": i + 1.0, "text": f"line number {i} here",
             "speaker": "Speaker 1" if i % 2 else "Speaker 2"} for i in range(100)]
    text = dialogue_lines(segs, {"Speaker 1": "Elena"}, 10, 60, keep={57.0}, limit=6)
    lines = text.splitlines()
    assert len(lines) == 6 and lines[0].startswith('- Someone said: "line number 10')  # unnamed
    assert '- Elena said: "line number 57 here"' in lines                             # kept
    assert all(10 <= int(l.split("number ")[1].split()[0]) < 60 for l in lines)


def test_clean_script_turns_screenplay_lines_into_narration():
    # Real outputs from the 8B model on the test drama.
    raw = ("[00:01:09] The wind began to shake the tower.\n"
           'Speaker 1: "Keep it steady, Elena." (Tomas)\n'
           'Elena Vasquez (Elena Vasquez): "The storm is over."\n'
           'Speaker 4: "It\'s turning away from the reef."\n'
           'He smiled. Speaker 2 chimes in with praise.')
    assert clean_script(raw).splitlines() == [
        "The wind began to shake the tower.",
        '"Keep it steady, Elena," Tomas said.',
        '"The storm is over," Elena Vasquez said.',
        '"It\'s turning away from the reef."',
        "He smiled. someone chimes in with praise.",
    ]
    normal = 'Tomas nodded. "For a storm like this, yes," he said.'
    assert clean_script(normal) == normal
    # Inline credit after a quote, and a description that only looks like a speaker.
    assert clean_script('She spoke up: "Wait. The lamp is flickering." (Elena Vasquez)') == \
        'She spoke up: "Wait. The lamp is flickering."'
    grim = "Tomas's expression turns grim: \"Light them and set them in front of the big mirror.\""
    assert clean_script(grim) == grim


def test_drop_repeats_removes_restated_ending():
    previous = "The storm broke over the tower. Elena saw the ship heading for the reef."
    text = "Elena saw the ship heading for the reef. The lamp began to flicker."
    assert drop_repeats(text, previous) == "The lamp began to flicker."


def test_story_parts_follow_length_and_resume(tmp_path):
    job = Job(tmp_path)
    backend = FakeBackend(n_ctx=3600)
    story = write_story(fake_segments(400), backend, job, "film", "suspense", "long", lambda f, m: None)
    assert backend.calls.count("part") == 8 and backend.calls.count("outline") == 1
    assert story["title"] == "The Long Night" and story["genre"] == "thriller"
    assert story["tone"] == "Suspense / thriller"
    assert [p["heading"] for p in story["parts"]][:2] == ["Chapter 1 Title", "Chapter 2 Title"]
    assert all("Here you go" not in p["text"] for p in story["parts"])

    again = FakeBackend(n_ctx=3600)
    write_story(fake_segments(400), again, job, "film", "suspense", "long", lambda f, m: None)
    assert again.calls == []  # beats, outline and every part come from story/*.json

    retone = FakeBackend(n_ctx=3600)
    write_story(fake_segments(400), retone, job, "film", "bedtime", "long", lambda f, m: None)
    assert retone.calls == ["part"] * 8  # a new tone rewrites the script only


def test_render_story():
    md = render_story({"title": "T", "logline": "L", "genre": "drama", "tone": "Emotional drama",
                       "characters": [{"name": "Ann", "description": "A pilot."}],
                       "parts": [{"heading": "Take-off", "text": "She flew."}], "words": 330},
                      "Retold from film.mp4")
    assert md.startswith("# T\n\n*L*") and "about 2 min read aloud" in md
    assert "- **Ann** — A pilot." in md and "## Take-off\n\nShe flew." in md
    assert "could not be generated" in render_story(None, "")
