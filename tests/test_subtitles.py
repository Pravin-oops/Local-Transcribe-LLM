import pytest

from localscribe.subtitles import SubtitleError, describe, load, parse_cues, to_segments

SRT = (
    "﻿1\r\n00:00:01,000 --> 00:00:03,000\r\n<i>Hello? Is anyone up here?</i>\r\n\r\n"
    "2\r\n00:00:04,000 --> 00:00:06,000\r\n- Who's there?\r\n- My name is Elena.\r\n\r\n"
    "3\r\n00:00:07,000 --> 00:00:09,500\r\nTOMAS: The radio is right.\r\nLook at those clouds.\r\n\r\n"
    "4\r\n00:00:10,000 --> 00:00:11,000\r\n[thunder rumbling]\r\n\r\n"
    "5\r\n00:00:12,000 --> 00:00:14,000\r\n{\\an8}(sighs) I know. ♪\r\n\r\n"
    "6\r\n00:00:15,000 --> 00:00:16,000\r\n♪ ♪\r\n"
)


def _write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def test_srt_cleanup_dash_dialogue_names_and_sound_cues(tmp_path):
    segs = load(_write(tmp_path, "film.srt", SRT))
    assert [(s["text"], s.get("speaker"), s.get("sound", False)) for s in segs] == [
        ("Hello? Is anyone up here?", None, False),
        ("Who's there?", None, False),
        ("My name is Elena.", None, False),
        ("The radio is right. Look at those clouds.", "Tomas", False),
        ("[thunder rumbling]", None, True),
        ("[sighs]", None, True),
        ("I know.", None, False),
    ]
    # Dash dialogue shares the cue's 2 s in proportion to text length, in order.
    who, name = segs[1], segs[2]
    assert who["start"] == 4.0 and who["end"] == name["start"] and name["end"] == 6.0
    assert who["end"] - who["start"] < name["end"] - name["start"]


def test_vtt_header_settings_and_short_times(tmp_path):
    vtt = ("WEBVTT\n\nNOTE made by hand\n\n"
           "intro\n00:01.500 --> 00:03.000 align:start position:10%\nFirst line\n\n"
           "01:00:00.000 --> 01:00:02.250\nAn hour in\n")
    segs = load(_write(tmp_path, "film.vtt", vtt))
    assert [(s["start"], s["end"], s["text"]) for s in segs] == [(1.5, 3.0, "First line"),
                                                                  (3600.0, 3602.25, "An hour in")]


def test_encodings(tmp_path):
    text = "1\n00:00:01,000 --> 00:00:02,000\nCafé crème, señor.\n"
    for enc in ("utf-16", "cp1252"):
        assert load(_write(tmp_path, f"x-{enc}.srt", text, enc))[0]["text"] == "Café crème, señor."


def test_mixed_case_label_is_not_a_speaker_and_bracket_names_are():
    segs = to_segments(parse_cues("1\n00:00:01,000 --> 00:00:02,000\nNote: this matters.\n\n"
                                  "2\n00:00:03,000 --> 00:00:04,000\n[MARA] Hold the line.\n"))
    assert segs[0].get("speaker") is None and segs[0]["text"] == "Note: this matters."
    assert segs[1] == {"start": 3.0, "end": 4.0, "text": "Hold the line.", "speaker": "Mara", "cue": True}


def test_each_cue_is_its_own_transcript_turn():
    from localscribe.merger import group_segments
    segs = to_segments(parse_cues("1\n00:00:01,000 --> 00:00:02,000\nWho's there?\n\n"
                                  "2\n00:00:02,100 --> 00:00:04,000\nMy name is Elena.\n"))
    # Audio segments this close together would merge; subtitle cues stay separate turns.
    assert [l["text"] for l in group_segments(segs)] == ["Who's there?", "My name is Elena."]


def test_errors_and_description(tmp_path):
    with pytest.raises(SubtitleError):
        load(_write(tmp_path, "empty.srt", "not a subtitle file"))
    with pytest.raises(SubtitleError):
        load(_write(tmp_path, "music.srt", "1\n00:00:01,000 --> 00:00:02,000\n[music playing]\n"))
    assert describe(_write(tmp_path, "film.srt", SRT)) == "5 lines · 00:00:14 · 1 named speaker · 2 sound cues"
