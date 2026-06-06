"""Speaker diarization and shared segmentation components."""

from src.diarization.clustering import (
    ahc_clustering,
    cluster_labels_to_speaker_segments,
    compute_overlap_matrix,
    compute_pseudo_der,
    map_clusters_to_speakers,
    sweep_threshold,
)
from src.diarization.embeddings import ECAPAEmbeddingConfig, ECAPAEmbeddingExtractor, l2_normalize
from src.diarization.scenario3 import ReferenceIdentifier, load_reference_embeddings
from src.diarization.scenarios import (
    run_scenario1_unknown_speakers,
    run_scenario2_oracle_speakers,
    run_scenario3_reference_identification,
)
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
    "ReferenceIdentifier",
    "ahc_clustering",
    "cluster_labels_to_speaker_segments",
    "compute_overlap_matrix",
    "compute_pseudo_der",
    "l2_normalize",
    "load_reference_embeddings",
    "map_clusters_to_speakers",
    "merge_speech_regions",
    "normalize_speech_regions",
    "prepare_asr_speech_regions",
    "prepare_embedding_segments",
    "run_scenario1_unknown_speakers",
    "run_scenario2_oracle_speakers",
    "run_scenario3_reference_identification",
    "split_long_speech_regions",
    "sweep_threshold",
]
