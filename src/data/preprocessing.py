"""Audio preprocessing utilities for AMI Meeting Corpus.

AMI audio is already 16kHz WAV, but may have multiple channels.
This module provides on-the-fly loading with resampling and mono conversion.
"""

import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def load_audio(
    audio_path: str | Path,
    target_sr: int = TARGET_SR,
    start_time: float | None = None,
    end_time: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio file with on-the-fly resampling and mono conversion.

    Processes one audio file at a time to manage memory efficiently.

    Args:
        audio_path: Path to the audio file (WAV).
        target_sr: Target sample rate (default: 16000).
        start_time: Optional start time in seconds for partial loading.
        end_time: Optional end time in seconds for partial loading.

    Returns:
        Tuple of (waveform, sample_rate) where waveform is shape (1, samples).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if start_time is not None or end_time is not None:
        waveform, sr = _load_audio_segment(audio_path, start_time, end_time)
    else:
        waveform, sr = torchaudio.load(audio_path)

    waveform = _to_mono(waveform)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr

    return waveform, sr


def _load_audio_segment(
    audio_path: Path,
    start_time: float | None,
    end_time: float | None,
) -> tuple[torch.Tensor, int]:
    """Load a specific time segment from an audio file.

    Args:
        audio_path: Path to the audio file.
        start_time: Start time in seconds.
        end_time: End time in seconds.

    Returns:
        Tuple of (waveform, sample_rate).
    """
    info = torchaudio.info(str(audio_path))
    sr = info.sample_rate

    frame_start = int(start_time * sr) if start_time is not None else 0
    if end_time is not None:
        num_frames = int((end_time - (start_time or 0)) * sr)
    else:
        num_frames = -1

    waveform, sr = torchaudio.load(
        audio_path,
        frame_offset=frame_start,
        num_frames=num_frames,
    )
    return waveform, sr


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert waveform to mono if it has multiple channels.

    Args:
        waveform: Input waveform of shape (channels, samples) or (1, samples).

    Returns:
        Mono waveform of shape (1, samples).
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def get_audio_duration(audio_path: str | Path) -> float:
    """Get the duration of an audio file in seconds without loading it.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds.
    """
    info = torchaudio.info(str(audio_path))
    return info.num_frames / info.sample_rate


def get_audio_info(audio_path: str | Path) -> dict:
    """Get detailed audio file information.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary with keys: sample_rate, num_channels, num_frames, duration.
    """
    info = torchaudio.info(str(audio_path))
    return {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration": info.num_frames / info.sample_rate,
    }
