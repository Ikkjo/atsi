import json

from src.integration.transcript import (
    build_integrated_transcript,
    build_transcript_segments,
    format_text_transcript,
    format_timestamp,
    refine_diarization_segments,
    save_json_transcript,
    save_text_transcript,
)


def test_build_transcript_segments_groups_adjacent_words_by_speaker():
    aligned_words = [
        {"start": 12.0, "end": 12.4, "word": "I", "speaker_id": "Speaker_A"},
        {"start": 12.4, "end": 13.0, "word": "agree", "speaker_id": "Speaker_A"},
        {"start": 15.0, "end": 15.5, "word": "But", "speaker_id": "Speaker_B"},
    ]

    segments = build_transcript_segments(aligned_words)

    assert segments[0]["start"] == 12.0
    assert segments[0]["end"] == 13.0
    assert segments[0]["speaker_id"] == "Speaker_A"
    assert segments[0]["text"] == "I agree"
    assert segments[1]["speaker_id"] == "Speaker_B"
    assert segments[1]["text"] == "But"


def test_format_text_transcript_matches_project_plan_shape():
    segments = [
        {"start": 12.0, "end": 15.9, "speaker_id": "Speaker_A", "text": "I think we need to change."},
        {"start": 15.9, "end": 20.0, "speaker_id": "Speaker_B", "text": "I agree."},
    ]

    assert format_text_transcript(segments) == (
        "[00:00:12 - 00:00:15] Speaker_A: I think we need to change.\n"
        "[00:00:15 - 00:00:20] Speaker_B: I agree."
    )


def test_build_integrated_transcript_includes_metadata_words_and_text():
    asr_result = {
        "meeting_id": "EN2001a",
        "audio_path": "/tmp/EN2001a.wav",
        "duration": 20.0,
        "model_id": "openai/whisper-small",
        "word_timestamp_mode": "native",
        "words": [
            {"start": 12.0, "end": 12.3, "word": "hello"},
            {"start": 16.0, "end": 16.5, "word": "there"},
        ],
    }
    diarization = [
        {"start": 10.0, "end": 15.0, "speaker_id": "Speaker_A"},
        {"start": 15.0, "end": 20.0, "speaker_id": "Speaker_B"},
    ]

    transcript = build_integrated_transcript(
        asr_result,
        diarization,
        scenario="scenario_2_oracle",
        microphone_configuration="ihm",
    )

    assert transcript["metadata"] == {
        "recording_name": "EN2001a",
        "scenario": "scenario_2_oracle",
        "microphone_configuration": "ihm",
        "asr_model_id": "openai/whisper-small",
        "word_timestamp_mode": "native",
        "duration": 20.0,
        "diarization_refinement": {
            "enabled": True,
            "min_segment_duration_s": 0.2,
            "max_merge_gap_s": 0.5,
            "input_segments": 2,
            "output_segments": 2,
        },
    }
    assert transcript["words"][0]["speaker_id"] == "Speaker_A"
    assert transcript["words"][1]["speaker_id"] == "Speaker_B"
    assert transcript["text"] == (
        "[00:00:12 - 00:00:12] Speaker_A: hello\n"
        "[00:00:16 - 00:00:16] Speaker_B: there"
    )


def test_refine_diarization_segments_filters_short_and_merges_adjacent():
    diarization = [
        {"start": 0.0, "end": 0.1, "speaker_id": "noise"},
        {"start": 1.0, "end": 1.5, "speaker_id": "Speaker_A"},
        {"start": 1.8, "end": 2.2, "speaker_id": "Speaker_A"},
        {"start": 3.0, "end": 4.0, "speaker_id": "Speaker_B"},
    ]

    refined = refine_diarization_segments(diarization)

    assert len(refined) == 2
    assert refined[0]["start"] == 1.0
    assert refined[0]["end"] == 2.2
    assert refined[0]["speaker_id"] == "Speaker_A"
    assert round(refined[0]["duration"], 3) == 1.2
    assert refined[1]["speaker_id"] == "Speaker_B"


def test_save_text_and_json_transcripts(tmp_path):
    transcript = {
        "metadata": {"recording_name": "M1"},
        "text": "[00:00:00 - 00:00:01] Speaker_A: hi",
        "segments": [],
        "words": [],
    }

    text_path = save_text_transcript(transcript, tmp_path / "nested" / "M1.txt")
    json_path = save_json_transcript(transcript, tmp_path / "nested" / "M1.json")

    assert text_path.read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] Speaker_A: hi\n"
    assert json.loads(json_path.read_text(encoding="utf-8")) == transcript


def test_format_timestamp_handles_none_and_hour_rollover():
    assert format_timestamp(None) == "--:--:--"
    assert format_timestamp(3661.9) == "01:01:01"
