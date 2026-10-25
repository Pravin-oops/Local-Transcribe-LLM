from localscribe.merger import group_segments, to_srt, to_txt, to_vtt

SEGS = [
    {"start": 0.0, "end": 4.0, "text": "Hello and welcome."},
    {"start": 4.5, "end": 9.0, "text": "Today we cover entropy."},
    {"start": 15.0, "end": 20.0, "text": "First, a definition."},
]


def test_group_segments_splits_on_long_pause():
    lines = group_segments(SEGS, max_span=30, max_gap=3)
    assert [l["text"] for l in lines] == [
        "Hello and welcome. Today we cover entropy.",
        "First, a definition.",
    ]
    assert SEGS[0]["text"] == "Hello and welcome."  # input not mutated


def test_group_segments_respects_max_span():
    lines = group_segments(SEGS, max_span=5, max_gap=10)
    assert len(lines) == 3


def test_srt_format():
    srt = to_srt(SEGS[:1])
    assert srt == "1\n00:00:00,000 --> 00:00:04,000\nHello and welcome.\n"


def test_vtt_and_txt():
    assert to_vtt(SEGS).startswith("WEBVTT\n\n00:00:00.000 --> 00:00:04.000\n")
    assert to_txt(SEGS).startswith("[00:00:00] Hello and welcome. Today we cover entropy.")
