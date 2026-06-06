"""Speaker diarization and shared segmentation components."""

from src.diarization.embeddings import ECAPAEmbeddingConfig, ECAPAEmbeddingExtractor, l2_normalize
from src.diarization.segmentation import (
    DiarizationSegmenter,
    EmbeddingSegmentationConfig,
    prepare_embedding_segments,
)
from src.diarization.vad import (
    PyannoteVAD,
    PyannoteVADConfig,
    merge_speech_regions,
    normalize_speech_regions,
    prepare_asr_speech_regions,
    split_long_speech_regions,
)

__all__ = [
    "DiarizationSegmenter",
    "ECAPAEmbeddingConfig",
    "ECAPAEmbeddingExtractor",
    "EmbeddingSegmentationConfig",
    "PyannoteVAD",
    "PyannoteVADConfig",
    "l2_normalize",
    "merge_speech_regions",
    "normalize_speech_regions",
    "prepare_asr_speech_regions",
    "prepare_embedding_segments",
    "split_long_speech_regions",
]
