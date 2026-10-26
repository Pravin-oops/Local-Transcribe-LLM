from localscribe.config import Settings
from localscribe.prompts import fields_for, map_schema


def test_presets_select_outputs_and_speakers():
    s = Settings()
    s.apply_preset("movie")
    assert (s.content_type, s.outputs, s.diarize) == ("movie", ["story"], True)
    s.apply_preset("lecture")
    assert (s.outputs, s.diarize) == (["summary", "notes"], False)


def test_v1_settings_file_still_loads():
    assert Settings.from_dict({"summarize": False, "whisper_model": "small.en"}).outputs == []
    assert Settings.from_dict({"summarize": True}).outputs == ["summary", "notes"]
    assert Settings.from_dict({"outputs": ["story", "bogus"]}).outputs == ["story"]


def test_map_fields_follow_outputs():
    assert fields_for(["summary"]) == ["chapters", "key_points"]
    assert fields_for(["notes"]) == ["key_points", "concepts", "examples"]
    assert fields_for(["story"]) == []
    assert map_schema(fields_for(["summary"]))["required"] == ["chapters", "key_points"]
