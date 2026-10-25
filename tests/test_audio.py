import numpy as np

from localscribe.audio import SAMPLE_RATE, plan_chunks

SR = SAMPLE_RATE


def _speech_with_gaps(total_s: int, gaps_at: list, gap_s: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(total_s * SR) * 8000).astype(np.int16)
    for t in gaps_at:
        x[int(t * SR): int((t + gap_s) * SR)] = 0
    return x


def test_chunks_cover_every_sample_once():
    x = _speech_with_gaps(600, [])
    chunks = plan_chunks(x, target_seconds=120)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(x)
    for (_, end), (start, _) in zip(chunks, chunks[1:]):
        assert end == start


def test_cuts_land_in_silence_near_target():
    gaps = [110, 235, 362]
    x = _speech_with_gaps(480, gaps)
    chunks = plan_chunks(x, target_seconds=120, search_seconds=30)
    cuts = [end / SR for _, end in chunks[:-1]]
    assert len(cuts) == 3
    for cut, gap in zip(cuts, gaps):
        assert gap <= cut <= gap + 1.0, (cut, gap)


def test_prefers_the_pause_nearest_the_target():
    # Two equally silent pauses inside the search window; the later one is closer to 120 s.
    x = _speech_with_gaps(300, [95, 118])
    first_cut = plan_chunks(x, target_seconds=120, search_seconds=30)[0][1] / SR
    assert 118 <= first_cut <= 119


def test_window_edges_are_not_mistaken_for_silence():
    x = _speech_with_gaps(300, [])  # no pause at all: any cut is fine, but not at the window edge
    first_cut = plan_chunks(x, target_seconds=120, search_seconds=30)[0][1] / SR
    assert 90.5 < first_cut < 149.5


def test_short_audio_is_one_chunk_and_no_tiny_tail():
    assert plan_chunks(_speech_with_gaps(90, []), target_seconds=120) == [[0, 90 * SR]]
    chunks = plan_chunks(_speech_with_gaps(260, []), target_seconds=120)
    assert all((end - start) / SR >= 60 for start, end in chunks)
