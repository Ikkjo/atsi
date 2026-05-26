"""Extract VAD-filtered reference embeddings for Scenario 3.

Extracts the first ~30 seconds of speech for each speaker from the
training/validation set, filters to speech-only regions using pyannote VAD,
and generates reference embeddings using ECAPA-TDNN (SpeechBrain).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model as PyannoteModel
from pyannote.audio.core.io import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
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
        self._embedding_model: EncoderClassifier | None = None
        self._audio = Audio(sample_rate=TARGET_SR, mono="downmix")

    @property
    def vad_model(self) -> PyannoteModel:
        """Lazy-load the pyannote VAD model."""
        if self._vad_model is None:
            logger.info("Loading pyannote VAD model...")
            self._vad_model = PyannoteModel.from_pretrained(
                "pyannote/segmentation-3.0",
                token=True,
            )
            self._vad_model.eval()
            self._vad_model.to(self.device)
            logger.info("VAD model loaded.")
        return self._vad_model

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
        segments = []

        try:
            binarize = Binarize(
                onset=0.5,
                offset=0.5,
                min_duration_on=0.1,
                min_duration_off=0.1,
            )

            for chunk, speech in self.vad_model(audio_path).itertracks(yield_label=True):
                speech_regions = binarize(speech)
                for region in speech_regions:
                    segments.append((region.start, region.end))
        except Exception as e:
            logger.warning("VAD failed for %s: %s", audio_path, e)

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
        applies VAD to filter to speech-only regions.

        Args:
            meeting_id: AMI meeting identifier.
            speaker_id: Speaker identifier.
            split: Split to use (train/validation).
            max_duration: Maximum total speech duration in seconds.

        Returns:
            List of waveform tensors (1, samples) at 16kHz.
        """
        segments = self.loader.get_meeting_segments(meeting_id, split)
        speaker_segments = [
            s for s in segments if s["speaker_id"] == speaker_id
        ]
        speaker_segments.sort(key=lambda s: s["begin_time"])

        audio_clips: list[torch.Tensor] = []
        total_duration = 0.0

        for seg in speaker_segments:
            if total_duration >= max_duration:
                break

            audio_path = seg["audio"]["path"]
            start = seg["begin_time"]
            end = seg["end_time"]

            try:
                waveform, sr = load_audio(audio_path, start_time=start, end_time=end)
            except Exception as e:
                logger.warning("Failed to load audio for %s: %s", seg.get("audio_id", "unknown"), e)
                continue

            speech_segments = self.get_speech_segments(audio_path)

            for speech_start, speech_end in speech_segments:
                seg_start = max(0, speech_start - start)
                seg_end = min(waveform.shape[1] / sr, speech_end - start)

                if seg_end <= seg_start:
                    continue

                start_frame = int(seg_start * sr)
                end_frame = int(seg_end * sr)
                clip = waveform[:, start_frame:end_frame]

                if clip.shape[1] < sr * 0.2:
                    continue

                audio_clips.append(clip)
                total_duration += (seg_end - seg_start)

                if total_duration >= max_duration:
                    break

        return audio_clips

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
