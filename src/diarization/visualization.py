"""Diarization visualization utilities.

Provides timeline plots for comparing predicted and ground-truth speaker
segmentations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

_DEFAULT_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _speaker_colour_map(speaker_ids: list[str], colours: list[str] | None = None) -> dict[str, str]:
    """Build a deterministic colour map for speaker IDs."""
    palette = colours or _DEFAULT_COLOURS
    unique = sorted(set(speaker_ids))
    return {
        spk: palette[i % len(palette)]
        for i, spk in enumerate(unique)
    }


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------


def plot_diarization_timeline(
    segments: list[dict[str, Any]],
    ax: Any | None = None,
    title: str = "Diarization",
    y_label: str = "Speaker",
    colour_map: dict[str, str] | None = None,
    colours: list[str] | None = None,
    show_legend: bool = True,
    xlim: tuple[float, float] | None = None,
) -> Any:
    """Plot a horizontal bar (Gantt-style) timeline of speaker segments.

    Args:
        segments: List of segment dicts with ``start``, ``end``, ``speaker_id``.
        ax: Optional matplotlib ``Axes`` object.  If ``None``, a new figure
            and axes are created via ``plt.subplots()``.
        title: Plot title.
        y_label: Y-axis label.
        colour_map: Optional pre-built speaker-to-colour mapping.  If omitted,
            one is generated from the unique speakers in *segments*.
        colours: Optional colour palette list (used if *colour_map* is omitted).
        show_legend: Whether to display a legend.
        xlim: Optional (xmin, xmax) tuple.  If omitted, computed from data.

    Returns:
        The matplotlib ``Axes`` object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, max(3, len({s["speaker_id"] for s in segments}) * 0.6)))

    if not segments:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        return ax

    if colour_map is None:
        colour_map = _speaker_colour_map(
            [seg["speaker_id"] for seg in segments], colours=colours
        )

    speakers = sorted({seg["speaker_id"] for seg in segments})
    speaker_to_y = {spk: i for i, spk in enumerate(speakers)}

    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        speaker = seg["speaker_id"]
        ax.barh(
            speaker_to_y[speaker],
            width=end - start,
            left=start,
            height=0.5,
            color=colour_map.get(speaker, "gray"),
            label=speaker if show_legend else "",
        )

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        all_starts = [float(s["start"]) for s in segments]
        all_ends = [float(s["end"]) for s in segments]
        pad = 0.5
        ax.set_xlim(max(0, min(all_starts) - pad), max(all_ends) + pad)

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    return ax


def plot_diarization_comparison(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    figsize: tuple[float, float] = (14, 6),
    title: str = "Diarization Comparison",
    colours: list[str] | None = None,
) -> Any:
    """Plot predicted vs ground-truth diarization timelines stacked vertically.

    Args:
        predicted: Predicted diarization segments.
        ground_truth: Ground-truth reference segments.
        figsize: Figure size in inches.
        title: Overall figure title.
        colours: Optional colour palette list.

    Returns:
        The matplotlib ``Figure`` object.
    """
    all_speakers = sorted({
        seg["speaker_id"]
        for seg in (predicted + ground_truth)
    })
    colour_map = _speaker_colour_map(all_speakers, colours=colours)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    plot_diarization_timeline(
        predicted,
        ax=axes[0],
        title="Predicted",
        colour_map=colour_map,
        show_legend=True,
    )
    plot_diarization_timeline(
        ground_truth,
        ax=axes[1],
        title="Ground Truth",
        colour_map=colour_map,
        show_legend=False,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_diarization_plot(
    segments: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Diarization",
    figsize: tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> Path:
    """Render a diarization timeline and save it to disk.

    Args:
        segments: Diarization segments.
        output_path: Output image path (``.png`` or ``.pdf``).
        title: Plot title.
        figsize: Figure size.
        dpi: Resolution for raster output.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    plot_diarization_timeline(segments, ax=ax, title=title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info("Saved diarization plot to %s", output_path)
    return output_path


def save_diarization_comparison_plot(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Diarization Comparison",
    figsize: tuple[float, float] = (14, 6),
    dpi: int = 150,
) -> Path:
    """Render a predicted-vs-ground-truth comparison and save it to disk.

    Args:
        predicted: Predicted segments.
        ground_truth: Ground-truth segments.
        output_path: Output image path.
        title: Figure title.
        figsize: Figure size.
        dpi: Resolution.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_diarization_comparison(predicted, ground_truth, figsize=figsize, title=title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    logger.info("Saved diarization comparison plot to %s", output_path)
    return output_path
