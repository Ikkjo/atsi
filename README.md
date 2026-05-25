# Project: Automatic Transcription with Speaker Identification

## Goal
Generate meeting audio transcriptions with timestamps and speaker identities, systematically comparing three speaker diarization modes on the AMI Meeting Corpus (IHM and SDM configurations).

## Tech Stack
Python, PyTorch, HuggingFace Transformers, pyannote.audio, SpeechBrain, HuggingFace Datasets, pyannote.metrics, jiwer, scipy

## Installation

```bash
# Create virtual environment with uv
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Hardware Notes

- CUDA is recommended but not required (falls back to CPU)
- If CUDA driver mismatch occurs, the pipeline will run on CPU (slower)
- Tune `whisper_batch_size` and `embedding_batch_size` based on available GPU memory

## Dataset
AMI Meeting Corpus via HuggingFace `datasets` (`edinburghcstr/ami` or `diarizers-community/ami`).

## License
MIT
