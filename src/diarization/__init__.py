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
from src.diarization.rttm import (
    filter_short_rttm_segments,
    merge_adjacent_rttm_segments,
    read_rttm,
    write_rttm,
)
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
from src.diarization.visualization import (
    plot_diarization_comparison,
    plot_diarization_timeline,
    save_diarization_comparison_plot,
    save_diarization_plot,
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
    "filter_short_rttm_segments",
    "l2_normalize",
    "load_reference_embeddings",
    "map_clusters_to_speakers",
    "merge_adjacent_rttm_segments",
    "merge_speech_regions",
    "normalize_speech_regions",
    "plot_diarization_comparison",
    "plot_diarization_timeline",
    "prepare_asr_speech_regions",
    "prepare_embedding_segments",
    "read_rttm",
    "run_scenario1_unknown_speakers",
    "run_scenario2_oracle_speakers",
    "run_scenario3_reference_identification",
    "save_diarization_comparison_plot",
    "save_diarization_plot",
    "split_long_speech_regions",
    "sweep_threshold",
    "write_rttm",
]
