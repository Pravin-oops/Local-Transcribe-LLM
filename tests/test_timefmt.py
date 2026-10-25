from localscribe.timefmt import hms, parse_hms, srt_time, vtt_time


def test_hms_truncates_and_pads():
    assert hms(0) == "00:00:00"
    assert hms(59.99) == "00:00:59"
    assert hms(3 * 3600 + 25 * 60 + 45.6) == "03:25:45"


def test_srt_and_vtt_round_milliseconds():
    assert srt_time(61.2346) == "00:01:01,235"
    assert vtt_time(3600.5) == "01:00:00.500"


def test_parse_hms_accepts_common_forms():
    assert parse_hms("01:02:03") == 3723
    assert parse_hms("2:03") == 123
    assert parse_hms("[00:12:30]") == 750
    assert parse_hms("00:12:30.5") == 750


def test_parse_hms_rejects_garbage():
    for bad in ("", "soon", "12", "00:75:00", None, 42):
        assert parse_hms(bad) is None
