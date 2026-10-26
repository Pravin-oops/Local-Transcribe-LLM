import numpy as np

from localscribe.diarizer import assign_speakers, merge_speakers
from localscribe.merger import group_segments, to_srt, to_txt


def test_merge_speakers_joins_same_voice_and_absorbs_fragments():
    # Each "voice" is a constant signal level; the fake embedding maps a level to a direction.
    sr = 16000
    samples = np.zeros(60 * sr, np.float32)
    turns = []
    for spk, (start, end, level) in enumerate([(0, 20, 0.1), (20, 40, 0.11), (40, 58, 0.5), (58, 59.5, 0.3)]):
        samples[int(start * sr):int(end * sr)] = level
        turns.append({"start": start, "end": end, "speaker": spk})
    directions = {0.1: [1, 0], 0.11: [0.99, 0.14], 0.5: [0, 1], 0.3: [0.6, 0.8]}
    embed = lambda clip: directions[round(float(clip[0]), 2)]
    merged = merge_speakers(samples, turns, embed)
    # 0 and 1 are the same voice (cos 0.99); 3 has only 1.5 s and leans to speaker 2 (cos 0.8 vs 0.6).
    assert [t["speaker"] for t in merged] == [0, 0, 2, 2]


def _seg(start, end, text="x"):
    return {"start": start, "end": end, "text": text}


def test_speakers_named_by_first_appearance_and_majority_overlap():
    turns = [{"start": 0, "end": 5, "speaker": 7}, {"start": 5, "end": 12, "speaker": 3}]
    out = assign_speakers([_seg(0, 4), _seg(4, 9), _seg(10, 11)], turns)
    # 4-9 overlaps speaker 7 for 1 s and speaker 3 for 4 s.
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 2", "Speaker 2"]


def test_gaps_use_nearest_turn_or_previous_speaker():
    turns = [{"start": 0, "end": 5, "speaker": 0}, {"start": 20, "end": 25, "speaker": 1}]
    out = assign_speakers([_seg(0, 2), _seg(5.5, 6.5), _seg(12, 13)], turns)
    assert [s["speaker"] for s in out] == ["Speaker 1", "Speaker 1", "Speaker 1"]
    assert assign_speakers([_seg(0, 1)], [])[0].get("speaker") is None


def test_transcript_formats_show_speakers_and_split_on_change():
    segs = [{"start": 0, "end": 2, "text": "Hi.", "speaker": "Speaker 1"},
            {"start": 2, "end": 4, "text": "Hello.", "speaker": "Speaker 2"},
            {"start": 4, "end": 6, "text": "How are you?", "speaker": "Speaker 2"}]
    assert len(group_segments(segs)) == 2
    assert "[00:00:00] Speaker 1: Hi.\n\n[00:00:02] Speaker 2: Hello. How are you?" in to_txt(segs)
    assert "Speaker 2: Hello." in to_srt(segs)
