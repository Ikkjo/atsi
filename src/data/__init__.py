"""Data loading, preprocessing, and annotation parsing for AMI Meeting Corpus."""

from typing import TYPE_CHECKING

from src.data.ami_loader import AMILoader
from src.data.preprocessing import load_audio
from src.data.annotation_parser import AnnotationParser
from src.data.split import SplitManager

if TYPE_CHECKING:
    from src.data.reference_embeddings import ReferenceEmbeddingExtractor

__all__ = [
    "AMILoader",
    "load_audio",
    "AnnotationParser",
    "SplitManager",
    "ReferenceEmbeddingExtractor",
]


def __getattr__(name: str):
    """Lazily import heavy pyannote/SpeechBrain reference extractor."""
    if name == "ReferenceEmbeddingExtractor":
        from src.data.reference_embeddings import ReferenceEmbeddingExtractor

        return ReferenceEmbeddingExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
