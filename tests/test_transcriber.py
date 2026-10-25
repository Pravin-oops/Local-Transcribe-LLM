from localscribe.transcriber import keep_segment


def test_drops_segment_stamped_past_chunk_end():
    # Seen on real audio: "Thank you." stamped 90.16-100.46 s in a 90.4 s chunk.
    assert not keep_segment(90.16, 100.46, "Thank you.", 90.4, "as latent heat.")


def test_keeps_last_real_segment_of_chunk():
    assert keep_segment(85.16, 90.16, "It is stored as latent heat.", 90.4, "Energy matters.")
    assert keep_segment(88.0, 90.6, "Slightly long end.", 90.4, None)  # small overrun is fine


def test_drops_exact_repeats_ignoring_case_and_punctuation():
    assert not keep_segment(298.6, 299.6, "Thank you, and see you next time.", 400,
                            "thank you and see you next time")
    assert keep_segment(10, 12, "See you next time.", 400, "Thank you.")


def test_drops_empty_text():
    assert not keep_segment(1, 2, " ... ", 400, None)
