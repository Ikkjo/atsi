"""Whisper ASR pipeline for meeting transcription.

This module wraps HuggingFace Transformers Whisper inference for AMI audio.
Audio is loaded through the local torchaudio preprocessing path instead of
letting Transformers decode files, avoiding torchcodec/FFmpeg issues.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.data.preprocessing import TARGET_SR, get_audio_duration, load_audio
from src.diarization.vad import normalize_speech_regions, prepare_asr_speech_regions
from src.utils.hardware import get_device, get_hardware_config

logger = logging.getLogger(__name__)

WordTimestampMode = Literal["native", "fallback", "off"]


@dataclass(frozen=True)
class WhisperASRConfig:
    """Configuration for Whisper inference."""

    model_id: str = "openai/whisper-large-v2"
    language: str = "english"
    task: str = "transcribe"
    target_sr: int = TARGET_SR
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    batch_size: int | None = None
    word_timestamp_mode: WordTimestampMode = "native"
    retry_on_cuda_oom: bool = True

    @classmethod
    def for_available_hardware(cls, **overrides: Any) -> "WhisperASRConfig":
        """Build a conservative config from available GPU memory."""
        hardware = get_hardware_config()
        total_memory_gb = hardware["total_memory_gb"]
        device = hardware["device"]

        if device == "cuda" and total_memory_gb > 20:
            chunk_length_s = 30.0
            stride_length_s = 5.0
        elif device == "cuda" and total_memory_gb > 10:
            chunk_length_s = 25.0
            stride_length_s = 4.0
        else:
            chunk_length_s = 20.0
            stride_length_s = 3.0

        values = {
            "batch_size": int(hardware["whisper_batch_size"]),
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
        }
        values.update(overrides)
        return cls(**values)


class WhisperASR:
    """Lazy-loading Whisper transcription pipeline.

    The default model follows the project plan (`openai/whisper-large-v2`). For
    quick iteration, pass a smaller model such as `openai/whisper-small`.
    """

    def __init__(self, config: WhisperASRConfig | None = None) -> None:
        self.config = config or WhisperASRConfig()
        self.device = get_device()
        self._pipeline = None

    @property
    def pipeline(self):
        """Create the HuggingFace ASR pipeline on first use."""
        if self._pipeline is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            device_arg = 0 if self.device.type == "cuda" else -1

            logger.info("Loading Whisper model: %s", self.config.model_id)
            processor = AutoProcessor.from_pretrained(self.config.model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device_arg,
            )
        return self._pipeline

    def transcribe(
        self,
        audio_path: str | Path,
        meeting_id: str | None = None,
        output_dir: str | Path | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Transcribe one audio file or time range.

        Args:
            audio_path: Path to an audio file.
            meeting_id: Optional AMI meeting ID for metadata and output naming.
            output_dir: Optional directory where the normalized result JSON is saved.
            start_time: Optional segment start in seconds.
            end_time: Optional segment end in seconds.
            use_cache: If True, return an existing saved result instead of re-running ASR.

        Returns:
            A JSON-serializable transcription record with utterance segments,
            word timestamps, and raw Whisper outputs.
        """
        audio_path = Path(audio_path)
        if use_cache and output_dir is not None:
            cache_path = self.result_path(audio_path, output_dir, meeting_id=meeting_id)
            if cache_path.exists():
                logger.info("Using cached Whisper result: %s", cache_path)
                return self.load_result(cache_path)

        audio_input = self._load_audio_input(audio_path, start_time=start_time, end_time=end_time)

        segment_raw = self._run_pipeline(audio_input, return_timestamps=True)
        segments = _extract_segments(segment_raw)
        words: list[dict[str, Any]] = []
        word_raw: dict[str, Any] | None = None
        word_mode_used: WordTimestampMode = "off"

        if self.config.word_timestamp_mode == "native":
            try:
                word_raw = self._run_pipeline(audio_input, return_timestamps="word")
                words = _extract_words(word_raw)
                word_mode_used = "native" if words else "fallback"
            except (TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Native Whisper word timestamps unavailable, using fallback: %s", exc)
                word_mode_used = "fallback"

        if self.config.word_timestamp_mode == "fallback" or word_mode_used == "fallback":
            words = _fallback_word_timestamps(segments)
            word_raw = word_raw or None
            word_mode_used = "fallback"

        duration = _audio_duration(audio_path, start_time=start_time, end_time=end_time)
        result: dict[str, Any] = {
            "meeting_id": meeting_id,
            "audio_path": str(audio_path),
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "sample_rate": self.config.target_sr,
            "model_id": self.config.model_id,
            "language": self.config.language,
            "task": self.config.task,
            "chunk_length_s": self.config.chunk_length_s,
            "stride_length_s": self.config.stride_length_s,
            "batch_size": self.batch_size,
            "device": self.device.type,
            "text": segment_raw.get("text", ""),
            "segments": segments,
            "words": words,
            "word_timestamp_mode": word_mode_used,
            "raw": {
                "segments": _json_safe(segment_raw),
                "words": _json_safe(word_raw) if word_raw is not None else None,
            },
        }

        if output_dir is not None:
            result["output_path"] = str(self.save_result(result, output_dir))

        return result

    def transcribe_batch(
        self,
        audio_paths: list[str | Path],
        output_dir: str | Path | None = None,
        meeting_ids: list[str] | None = None,
        use_cache: bool = False,
    ) -> list[dict[str, Any]]:
        """Transcribe multiple recordings one at a time.

        Whisper internally batches model chunks via `batch_size`; this wrapper
        intentionally avoids loading multiple long meetings into RAM at once.
        """
        if meeting_ids is not None and len(meeting_ids) != len(audio_paths):
            raise ValueError("meeting_ids must have the same length as audio_paths")

        results = []
        for idx, audio_path in enumerate(audio_paths):
            meeting_id = meeting_ids[idx] if meeting_ids else None
            results.append(
                self.transcribe(
                    audio_path,
                    meeting_id=meeting_id,
                    output_dir=output_dir,
                    use_cache=use_cache,
                )
            )
        return results

    def transcribe_speech_regions(
        self,
        audio_path: str | Path,
        speech_regions: list[dict[str, Any]],
        meeting_id: str | None = None,
        output_dir: str | Path | None = None,
        use_cache: bool = False,
        merge_gap_s: float = 0.5,
        max_region_duration_s: float | None = None,
    ) -> dict[str, Any]:
        """Transcribe precomputed VAD speech regions and preserve absolute timestamps.

        VAD detection remains in `src.diarization`; ASR only consumes the shared
        region format when full-meeting Whisper transcription is not desirable.
        """
        audio_path = Path(audio_path)
        if use_cache and output_dir is not None:
            cache_path = self.result_path(audio_path, output_dir, meeting_id=meeting_id)
            if cache_path.exists():
                logger.info("Using cached Whisper result: %s", cache_path)
                return self.load_result(cache_path)

        regions = prepare_asr_speech_regions(
            speech_regions,
            merge_gap_s=merge_gap_s,
            max_duration_s=max_region_duration_s,
        )

        region_results = []
        segments: list[dict[str, Any]] = []
        words: list[dict[str, Any]] = []
        text_parts = []

        for region_idx, region in enumerate(regions):
            region_result = self.transcribe(
                audio_path,
                start_time=float(region["start"]),
                end_time=float(region["end"]),
                use_cache=False,
            )
            region_results.append(_region_summary(region_result, region_idx))
            text = (region_result.get("text") or "").strip()
            if text:
                text_parts.append(text)
            segments.extend(_offset_items(region_result.get("segments") or [], float(region["start"]), len(segments)))
            words.extend(_offset_items(region_result.get("words") or [], float(region["start"]), len(words)))

        result: dict[str, Any] = {
            "meeting_id": meeting_id,
            "audio_path": str(audio_path),
            "start_time": None,
            "end_time": None,
            "duration": get_audio_duration(audio_path),
            "sample_rate": self.config.target_sr,
            "model_id": self.config.model_id,
            "language": self.config.language,
            "task": self.config.task,
            "chunk_length_s": self.config.chunk_length_s,
            "stride_length_s": self.config.stride_length_s,
            "batch_size": self.batch_size,
            "device": self.device.type,
            "text": " ".join(text_parts).strip(),
            "segments": segments,
            "words": words,
            "word_timestamp_mode": _combined_word_mode(region_results),
            "vad": {
                "source": "precomputed_speech_regions",
                "merge_gap_s": merge_gap_s,
                "max_region_duration_s": max_region_duration_s,
                "input_regions": normalize_speech_regions(speech_regions),
                "transcribed_regions": regions,
            },
            "raw": {
                "region_results": region_results,
            },
        }

        if output_dir is not None:
            result["output_path"] = str(self.save_result(result, output_dir))

        return result

    @property
    def batch_size(self) -> int:
        """Return configured or hardware-derived Whisper batch size."""
        if self.config.batch_size is not None:
            return self.config.batch_size
        return int(get_hardware_config()["whisper_batch_size"])

    def save_result(self, result: dict[str, Any], output_dir: str | Path) -> Path:
        """Save a transcription record as JSON."""
        output_path = self.result_path(result["audio_path"], output_dir, meeting_id=result.get("meeting_id"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(_json_safe(result), f, indent=2)
        return output_path

    @staticmethod
    def load_result(path: str | Path) -> dict[str, Any]:
        """Load a saved Whisper transcription JSON."""
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def result_path(
        audio_path: str | Path,
        output_dir: str | Path,
        meeting_id: str | None = None,
    ) -> Path:
        """Return the deterministic cache path for a transcription result."""
        stem = meeting_id or Path(audio_path).stem
        return Path(output_dir) / f"{stem}_whisper.json"

    def _run_pipeline(self, audio_input: dict[str, Any], return_timestamps: bool | str) -> dict[str, Any]:
        return self._run_pipeline_with_batch(audio_input, return_timestamps, self.batch_size)

    def _run_pipeline_with_batch(
        self,
        audio_input: dict[str, Any],
        return_timestamps: bool | str,
        batch_size: int,
    ) -> dict[str, Any]:
        generate_kwargs = {
            "language": self.config.language,
            "task": self.config.task,
        }
        try:
            output = self.pipeline(
                audio_input,
                return_timestamps=return_timestamps,
                chunk_length_s=self.config.chunk_length_s,
                stride_length_s=self.config.stride_length_s,
                batch_size=batch_size,
                generate_kwargs=generate_kwargs,
            )
        except RuntimeError as exc:
            if not self.config.retry_on_cuda_oom or batch_size <= 1 or not _is_cuda_oom(exc):
                raise
            logger.warning("Whisper CUDA OOM at batch_size=%s; retrying with batch_size=1", batch_size)
            torch.cuda.empty_cache()
            output = self.pipeline(
                audio_input,
                return_timestamps=return_timestamps,
                chunk_length_s=self.config.chunk_length_s,
                stride_length_s=self.config.stride_length_s,
                batch_size=1,
                generate_kwargs=generate_kwargs,
            )
        return dict(output)

    def _load_audio_input(
        self,
        audio_path: Path,
        start_time: float | None,
        end_time: float | None,
    ) -> dict[str, Any]:
        waveform, sample_rate = load_audio(
            audio_path,
            target_sr=self.config.target_sr,
            start_time=start_time,
            end_time=end_time,
        )
        return {
            "raw": waveform.squeeze(0).cpu().numpy(),
            "sampling_rate": sample_rate,
        }


def _audio_duration(audio_path: Path, start_time: float | None, end_time: float | None) -> float:
    if start_time is not None or end_time is not None:
        full_duration = get_audio_duration(audio_path)
        start = start_time or 0.0
        end = end_time if end_time is not None else full_duration
        return max(0.0, end - start)
    return get_audio_duration(audio_path)


def _offset_items(items: list[dict[str, Any]], offset_s: float, start_id: int) -> list[dict[str, Any]]:
    """Offset region-local timestamp records to absolute meeting time."""
    offset = []
    for idx, item in enumerate(items):
        shifted = dict(item)
        shifted["id"] = start_id + idx
        if shifted.get("start") is not None:
            shifted["start"] = float(shifted["start"]) + offset_s
        if shifted.get("end") is not None:
            shifted["end"] = float(shifted["end"]) + offset_s
        offset.append(shifted)
    return offset


def _region_summary(result: dict[str, Any], region_idx: int) -> dict[str, Any]:
    """Keep enough per-region raw metadata to audit VAD-based ASR output."""
    return {
        "region_index": region_idx,
        "start_time": result.get("start_time"),
        "end_time": result.get("end_time"),
        "text": result.get("text", ""),
        "word_timestamp_mode": result.get("word_timestamp_mode"),
        "segments": result.get("segments", []),
        "words": result.get("words", []),
    }


def _combined_word_mode(region_results: list[dict[str, Any]]) -> WordTimestampMode:
    modes = {result.get("word_timestamp_mode") for result in region_results}
    if not modes:
        return "off"
    if modes == {"native"}:
        return "native"
    if modes <= {"fallback", "native"}:
        return "fallback"
    if modes == {"off"}:
        return "off"
    return "fallback"


def _extract_segments(raw_output: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = raw_output.get("chunks") or []
    if not chunks:
        return []

    segments = []
    for idx, chunk in enumerate(chunks):
        start, end = _parse_timestamp(chunk.get("timestamp"))
        segments.append(
            {
                "id": idx,
                "start": start,
                "end": end,
                "text": (chunk.get("text") or "").strip(),
            }
        )
    return segments


def _extract_words(raw_output: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = raw_output.get("chunks") or []
    words = []
    for idx, chunk in enumerate(chunks):
        start, end = _parse_timestamp(chunk.get("timestamp"))
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        words.append(
            {
                "id": idx,
                "start": start,
                "end": end,
                "word": text,
                "source": "native",
            }
        )
    return words


def _fallback_word_timestamps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Estimate word timestamps by proportional character length per segment."""
    words = []
    word_id = 0

    for segment in segments:
        tokens = _tokenize_words(segment["text"])
        start = segment.get("start")
        end = segment.get("end")
        if not tokens or start is None or end is None or end <= start:
            continue

        total_chars = sum(max(1, len(token)) for token in tokens)
        cursor = float(start)
        duration = float(end) - float(start)

        for token in tokens:
            token_duration = duration * (max(1, len(token)) / total_chars)
            token_end = min(float(end), cursor + token_duration)
            words.append(
                {
                    "id": word_id,
                    "start": cursor,
                    "end": token_end,
                    "word": token,
                    "source": "fallback_char_proportional",
                }
            )
            word_id += 1
            cursor = token_end

    return words


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())


def _parse_timestamp(timestamp: Any) -> tuple[float | None, float | None]:
    if timestamp is None:
        return None, None
    if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
        return _to_float_or_none(timestamp[0]), _to_float_or_none(timestamp[1])
    return None, None


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value
