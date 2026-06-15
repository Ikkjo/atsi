"""Extract VAD-filtered reference embeddings for Scenario 3.

Extracts the first ~30 seconds of speech for each speaker from the
training/validation set, filters to speech-only regions using pyannote VAD,
and generates reference embeddings using ECAPA-TDNN (SpeechBrain).
"""

import logging
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Inference, Model as PyannoteModel
from pyannote.audio.utils.signal import Binarize
from speechbrain.inference.classifiers import EncoderClassifier

from src.data.ami_loader import AMILoader
from src.data.preprocessing import load_audio
from src.utils.hardware import get_device

logger = logging.getLogger(__name__)

TARGET_SR = 16_000
REFERENCE_DURATION = 30.0
EMBEDDING_DIM = 192


class ReferenceEmbeddingExtractor:
    """Extract VAD-filtered reference speaker embeddings.

    For each speaker in a meeting, extracts clean speech segments (filtered
    by VAD), computes ECAPA-TDNN embeddings, and averages them into a
    single L2-normalized reference embedding.

    Args:
        loader: AMILoader instance for accessing dataset.
        device: Torch device for model inference.
    """

    def __init__(
        self,
        loader: AMILoader,
        device: torch.device | None = None,
    ) -> None:
        self.loader = loader
        self.device = device or get_device()
        self._vad_model: PyannoteModel | None = None
        self._vad_inference: Inference | None = None
        self._embedding_model: EncoderClassifier | None = None
        self._vad_cache: dict[str, list[tuple[float, float]]] = {}

    @property
    def vad_model(self) -> PyannoteModel | None:
        """Lazy-load the pyannote VAD model.

        Returns None if the model cannot be loaded (e.g. due to a
        ``huggingface_hub`` version mismatch with ``pyannote-audio``).
        Callers should check for None and degrade gracefully.
        """
        if self._vad_model is None:
            logger.info("Loading pyannote VAD model...")
            try:
                self._vad_model = PyannoteModel.from_pretrained(
                    "pyannote/segmentation-3.0",
                    token=True,
                )
                self._vad_model.eval()
                self._vad_model.to(self.device)
                logger.info("VAD model loaded.")
            except Exception as e:
                logger.warning(
                    "Failed to load pyannote VAD model: %s. "
                    "VAD will be skipped; annotation-only intervals will be used.",
                    e,
                )
                self._vad_model = None
        return self._vad_model

    @property
    def vad_inference(self) -> Inference | None:
        """Lazy-load pyannote inference wrapper for the segmentation model.

        Returns None if the VAD model could not be loaded.
        """
        if self._vad_inference is None:
            model = self.vad_model
            if model is None:
                logger.warning("VAD model not available; inference disabled.")
                return None
            self._vad_inference = Inference(model, device=self.device)
        return self._vad_inference

    @property
    def embedding_model(self) -> EncoderClassifier:
        """Lazy-load the SpeechBrain ECAPA-TDNN embedding model."""
        if self._embedding_model is None:
            logger.info("Loading ECAPA-TDNN embedding model...")
            self._embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(self.device)},
            )
            logger.info("ECAPA-TDNN model loaded.")
        return self._embedding_model

    def get_speech_segments(self, audio_path: str | Path) -> list[tuple[float, float]]:
        """Detect speech segments using pyannote VAD model.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of (start, end) tuples in seconds.
        """
        audio_path = str(audio_path)
        if audio_path in self._vad_cache:
            return self._vad_cache[audio_path]

        segments: list[tuple[float, float]] = []

        infer = self.vad_inference
        if infer is not None:
            try:
                binarize = Binarize(
                    onset=0.5,
                    offset=0.5,
                    min_duration_on=0.1,
                    min_duration_off=0.1,
                )

                prediction = infer(audio_path)
                speech_annotation = binarize(prediction)
                for region in speech_annotation.get_timeline().support():
                    segments.append((region.start, region.end))
            except Exception as e:
                logger.warning("VAD failed for %s: %s", audio_path, e)

        self._vad_cache[audio_path] = segments
        return segments

    def extract_speaker_speech(
        self,
        meeting_id: str,
        speaker_id: str,
        split: str = "train",
        max_duration: float = REFERENCE_DURATION,
    ) -> list[torch.Tensor]:
        """Extract speech-only audio segments for a specific speaker.

        Uses ground truth annotations to identify speaker segments, then
        applies VAD and removes regions that overlap with other annotated
        speakers. If VAD fails, falls back to annotation-only regions so the
        caller can still generate a clearly logged baseline reference.

        Args:
            meeting_id: AMI meeting identifier.
            speaker_id: Speaker identifier.
            split: Split to use (train/validation).
            max_duration: Maximum total speech duration in seconds.

        Returns:
            List of waveform tensors (1, samples) at 16kHz.
        """
        from src.data.meeting_audio import get_meeting_audio_path

        segments = self.loader.get_meeting_segments(meeting_id, split)
        speaker_segments = [s for s in segments if s["speaker_id"] == speaker_id]
        speaker_segments.sort(key=lambda s: s["begin_time"])
        other_speaker_intervals = [
            (s["begin_time"], s["end_time"])
            for s in segments
            if s["speaker_id"] != speaker_id
        ]

        # Get the full meeting audio path (reconstructed if needed)
        audio_path = get_meeting_audio_path(self.loader, meeting_id, split)
        # Run VAD once on the full meeting audio
        speech_segments = self.get_speech_segments(audio_path)

        audio_clips: list[torch.Tensor] = []
        total_duration = 0.0

        for seg in speaker_segments:
            if total_duration >= max_duration:
                break

            start = seg["begin_time"]
            end = seg["end_time"]

            candidate_intervals = self._intersect_intervals(
                [(start, end)],
                speech_segments,
            )
            if not candidate_intervals:
                logger.warning(
                    "No VAD speech overlap for %s/%s %.2f-%.2f; using annotation interval",
                    meeting_id,
                    speaker_id,
                    start,
                    end,
                )
                candidate_intervals = [(start, end)]

            clean_intervals = self._subtract_intervals(
                candidate_intervals,
                other_speaker_intervals,
            )

            for clean_start, clean_end in clean_intervals:
                if clean_end <= clean_start:
                    continue

                try:
                    clip, sr = load_audio(
                        audio_path,
                        start_time=clean_start,
                        end_time=clean_end,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to load audio for %s: %s",
                        seg.get("audio_id", "unknown"),
                        e,
                    )
                    continue

                if clip.shape[1] < sr * 0.2:
                    continue

                audio_clips.append(clip)
                total_duration += (clean_end - clean_start)

                if total_duration >= max_duration:
                    break

        return audio_clips

    @staticmethod
    def _intersect_intervals(
        base_intervals: list[tuple[float, float]],
        filter_intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Return intersections between two interval lists."""
        intersections: list[tuple[float, float]] = []
        for base_start, base_end in base_intervals:
            for filter_start, filter_end in filter_intervals:
                start = max(base_start, filter_start)
                end = min(base_end, filter_end)
                if end > start:
                    intersections.append((start, end))
        return intersections

    @staticmethod
    def _subtract_intervals(
        intervals: list[tuple[float, float]],
        blocked: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Remove blocked intervals, preserving non-overlapping speech regions."""
        remaining = intervals[:]
        for block_start, block_end in sorted(blocked):
            next_remaining: list[tuple[float, float]] = []
            for start, end in remaining:
                if block_end <= start or block_start >= end:
                    next_remaining.append((start, end))
                    continue
                if block_start > start:
                    next_remaining.append((start, block_start))
                if block_end < end:
                    next_remaining.append((block_end, end))
            remaining = next_remaining
        return remaining

    def compute_embedding(self, waveform: torch.Tensor) -> np.ndarray:
        """Compute ECAPA-TDNN embedding for a waveform.

        Args:
            waveform: Waveform tensor of shape (1, samples).

        Returns:
            Embedding array of shape (192,).
        """
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        with torch.no_grad():
            embedding = self.embedding_model.encode_batch(waveform)
            embedding = embedding.squeeze(0).squeeze(0).cpu().numpy()

        return embedding

    def compute_reference_embedding(
        self,
        meeting_id: str,
        speaker_id: str,
        split: str = "train",
    ) -> np.ndarray:
        """Compute averaged, L2-normalized reference embedding for a speaker.

        Args:
            meeting_id: AMI meeting identifier.
            speaker_id: Speaker identifier.
            split: Split to use.

        Returns:
            L2-normalized embedding of shape (192,).
        """
        clips = self.extract_speaker_speech(meeting_id, speaker_id, split)

        if not clips:
            raise ValueError(
                f"No speech clips found for speaker {speaker_id} in meeting {meeting_id}"
            )

        embeddings = []
        for clip in clips:
            emb = self.compute_embedding(clip)
            embeddings.append(emb)

        avg_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    def save_reference_embedding(
        self,
        meeting_id: str,
        speaker_id: str,
        split: str = "train",
        output_dir: str = "data/references",
    ) -> Path:
        """Compute and save reference embedding for a speaker.

        Args:
            meeting_id: AMI meeting identifier.
            speaker_id: Speaker identifier.
            split: Split to use.
            output_dir: Directory to save the .pt file.

        Returns:
            Path to the saved .pt file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        embedding = self.compute_reference_embedding(meeting_id, speaker_id, split)

        output_path = out / f"{meeting_id}_{speaker_id}.pt"
        torch.save(
            {
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "embedding": torch.from_numpy(embedding),
                "dimension": EMBEDDING_DIM,
                "normalized": True,
            },
            output_path,
        )

        logger.info("Saved reference embedding to %s", output_path)
        return output_path

    def save_all_references(
        self,
        split: str = "train",
        output_dir: str = "data/references",
        meeting_ids: list[str] | None = None,
    ) -> list[Path]:
        """Compute and save reference embeddings for all speakers in a split.

        Args:
            split: Split to use.
            output_dir: Directory to save .pt files.
            meeting_ids: Optional list of meeting IDs to process.

        Returns:
            List of paths to saved .pt files.
        """
        if meeting_ids is None:
            meeting_ids = self.loader.get_meeting_ids(split)

        paths = []
        for meeting_id in meeting_ids:
            speakers = self.loader.get_meeting_speakers(meeting_id, split)
            for speaker_id in speakers:
                try:
                    path = self.save_reference_embedding(
                        meeting_id, speaker_id, split, output_dir
                    )
                    paths.append(path)
                except Exception as e:
                    logger.error(
                        "Failed to extract reference for %s/%s: %s",
                        meeting_id,
                        speaker_id,
                        e,
                    )

        return paths

    def save_meeting_references(
        self,
        meeting_id: str,
        split: str = "train",
        output_dir: str = "data/references",
    ) -> list[Path]:
        """Compute and save references for all speakers in one meeting.

        This is the preferred Scenario 3 enrollment entry point: references
        are generated for the same meeting whose speakers will be identified.
        """
        paths = []
        for speaker_id in self.loader.get_meeting_speakers(meeting_id, split):
            paths.append(
                self.save_reference_embedding(meeting_id, speaker_id, split, output_dir)
            )
        return paths
