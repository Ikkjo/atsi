import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # Non-interactive backend for headless tests

from src.diarization.visualization import (
    plot_diarization_comparison,
    plot_diarization_timeline,
    save_diarization_comparison_plot,
    save_diarization_plot,
)


def test_plot_diarization_timeline_basic():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.5, "end": 3.0, "speaker_id": "B"},
        {"start": 3.5, "end": 4.0, "speaker_id": "A"},
    ]
    ax = plot_diarization_timeline(segments, title="Test")
    assert ax.get_title() == "Test"
    plt.close(ax.figure)


def test_plot_diarization_timeline_empty():
    ax = plot_diarization_timeline([], title="Empty")
    assert ax.get_title() == "Empty"
    plt.close(ax.figure)


def test_plot_diarization_timeline_with_existing_ax():
    fig, ax = plt.subplots()
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
    ]
    returned = plot_diarization_timeline(segments, ax=ax, title="Existing")
    assert returned is ax
    plt.close(fig)


def test_plot_diarization_timeline_xlim():
    segments = [
        {"start": 5.0, "end": 6.0, "speaker_id": "A"},
    ]
    ax = plot_diarization_timeline(segments, xlim=(0, 10))
    xmin, xmax = ax.get_xlim()
    assert xmin == 0.0
    assert xmax == 10.0
    plt.close(ax.figure)


def test_plot_diarization_comparison():
    predicted = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "B"},
    ]
    ground_truth = [
        {"start": 0.0, "end": 1.2, "speaker_id": "A"},
        {"start": 1.2, "end": 2.0, "speaker_id": "B"},
    ]
    fig = plot_diarization_comparison(predicted, ground_truth, title="Compare")
    assert fig._suptitle.get_text() == "Compare"
    plt.close(fig)


def test_save_diarization_plot(tmp_path):
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
    ]
    path = tmp_path / "timeline.png"
    result = save_diarization_plot(segments, path, title="Saved")
    assert result.exists()
    assert result.suffix == ".png"


def test_save_diarization_comparison_plot(tmp_path):
    predicted = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
    ]
    ground_truth = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
    ]
    path = tmp_path / "comparison.png"
    result = save_diarization_comparison_plot(predicted, ground_truth, path, title="Comp")
    assert result.exists()
    assert result.suffix == ".png"
