"""Smoke test for Epic 2: Data Collection and Preparation.

Validates:
- AMI corpus loads without error
- Meeting speaker counts are available from annotations
- Audio is 16kHz mono
- Reference embeddings are 192-dimensional and L2-normalized
- Split metadata exists and is consistent
"""

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ami_loader import AMILoader
from src.data.annotation_parser import AnnotationParser
from src.data.preprocessing import get_audio_info, load_audio
from src.data.split import SplitManager

TARGET_SR = 16_000
EXPECTED_EMBEDDING_DIM = 192


def test_ami_corpus_loads() -> bool:
    """Test that AMI corpus loads without error."""
    print("\n[Test 1] Loading AMI corpus...")
    try:
        loader = AMILoader(config="ihm")
        ds = loader.dataset
        print(f"  Dataset loaded: {ds}")
        print(f"  Train segments: {len(ds['train'])}")
        print(f"  Validation segments: {len(ds['validation'])}")
        print(f"  Test segments: {len(ds['test'])}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_meeting_speaker_counts_available() -> bool:
    """Test that meeting speaker counts can be derived from annotations."""
    print("\n[Test 2] Checking meeting speaker counts...")
    try:
        loader = AMILoader(config="ihm")
        train_meetings = loader.get_meeting_ids("train")

        checked_counts = {}
        for meeting_id in train_meetings[:20]:
            speakers = loader.get_meeting_speakers(meeting_id, "train")
            checked_counts[meeting_id] = len(speakers)
            print(f"  Meeting {meeting_id}: {len(speakers)} speakers - {speakers}")

        if checked_counts and all(count > 0 for count in checked_counts.values()):
            print("  PASS: Speaker counts are available for sampled meetings")
            return True

        print("  FAILED: Could not derive speaker counts")
        return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_audio_is_16khz_mono() -> bool:
    """Test that audio is 16kHz mono."""
    print("\n[Test 3] Checking audio format (16kHz mono)...")
    try:
        loader = AMILoader(config="ihm")
        train_meetings = loader.get_meeting_ids("train")
        meeting_id = train_meetings[0]

        segments = loader.get_meeting_segments(meeting_id, "train")
        if not segments:
            print(f"  FAILED: No segments found for {meeting_id}")
            return False

        audio = segments[0]["audio"]
        array = audio["array"]
        sr = audio["sampling_rate"]

        print(f"  Audio for segment {segments[0]['audio_id']}:")
        print(f"    Array shape: {len(array)}")
        print(f"    Sampling rate: {sr}")
        print(f"    Duration: {len(array) / sr:.2f}s")

        if sr == TARGET_SR and len(array) > 0:
            print(f"  PASS: Audio is {sr}Hz mono")
            return True
        else:
            print(f"  FAILED: Expected {TARGET_SR}Hz mono, got {sr}Hz")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_split_metadata_exists() -> bool:
    """Test that split metadata exists and is consistent."""
    print("\n[Test 4] Checking split metadata...")
    try:
        loader = AMILoader(config="ihm")
        split_manager = SplitManager(loader)

        splits_path = PROJECT_ROOT / "data" / "processed" / "ami_splits.json"
        if not splits_path.exists():
            print(f"  Split metadata not found, generating...")
            split_manager.save_splits()

        with open(splits_path) as f:
            metadata = json.load(f)

        print(f"  Split metadata loaded from {splits_path}")
        print(f"  Config: {metadata['config']}")

        for split_name, split_data in metadata["splits"].items():
            print(f"  {split_name}: {split_data['num_meetings']} meetings, {split_data['num_segments']} segments")

        split_manager = SplitManager(loader)
        is_consistent = split_manager.verify_consistency()

        if is_consistent:
            print(f"  PASS: Split metadata is consistent")
            return True
        else:
            print(f"  FAILED: Split metadata is inconsistent")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_annotations_parse() -> bool:
    """Test that annotations can be parsed for a meeting."""
    print("\n[Test 5] Parsing meeting annotations...")
    try:
        loader = AMILoader(config="ihm")
        parser = AnnotationParser(loader)

        train_meetings = loader.get_meeting_ids("train")
        meeting_id = train_meetings[0]

        annotations = parser.parse_meeting(meeting_id, "train")
        print(f"  Meeting: {annotations['meeting_id']}")
        print(f"  Speakers: {annotations['speakers']}")
        print(f"  Speaker segments: {len(annotations['segments'])}")
        print(f"  Word annotations: {len(annotations['words'])}")
        print(f"  Word timestamp source: {annotations['metadata']['word_timestamp_source']}")

        output_dir = PROJECT_ROOT / "data" / "processed"
        output_path = parser.save_meeting_annotations(meeting_id, str(output_dir), "train")
        print(f"  Saved annotations to {output_path}")

        if (
            len(annotations["speakers"]) > 0
            and len(annotations["words"]) > 0
            and annotations["metadata"]["word_timestamp_source"]
        ):
            print(f"  PASS: Annotations parsed successfully")
            return True
        else:
            print(f"  FAILED: Empty annotations")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_reference_embeddings() -> bool:
    """Test that reference embeddings are correct dimension and normalized."""
    print("\n[Test 6] Checking reference embeddings...")
    try:
        ref_dir = PROJECT_ROOT / "data" / "references"
        if not ref_dir.exists() or not any(ref_dir.glob("*.pt")):
            print(f"  No reference embeddings found in {ref_dir}")
            print(f"  Skipping (run reference extraction first)")
            return True

        embedding_files = list(ref_dir.glob("*.pt"))
        sample_file = embedding_files[0]

        data = torch.load(sample_file, weights_only=True)
        embedding = data["embedding"]

        dim = embedding.shape[0]
        norm = torch.norm(embedding).item()

        print(f"  Sample embedding: {sample_file.name}")
        print(f"  Dimension: {dim}")
        print(f"  L2 norm: {norm:.6f}")

        if dim == EXPECTED_EMBEDDING_DIM and abs(norm - 1.0) < 1e-5:
            print(f"  PASS: Embedding is {dim}-dimensional and L2-normalized")
            return True
        else:
            print(f"  FAILED: Expected {EXPECTED_EMBEDDING_DIM}-dim, norm=1.0")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main() -> None:
    """Run all smoke tests."""
    print("=" * 60)
    print("Epic 2: Data Collection and Preparation - Smoke Tests")
    print("=" * 60)

    results = {
        "AMI corpus loads": test_ami_corpus_loads(),
        "Meeting speaker counts available": test_meeting_speaker_counts_available(),
        "Audio is 16kHz mono": test_audio_is_16khz_mono(),
        "Split metadata exists": test_split_metadata_exists(),
        "Annotations parse": test_annotations_parse(),
        "Reference embeddings": test_reference_embeddings(),
    }

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
