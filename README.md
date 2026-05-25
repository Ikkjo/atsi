# Project: Automatic Transcription with Speaker Identification

## Goal
Generate meeting audio transcriptions with timestamps and speaker identities, systematically comparing three speaker diarization modes on the AMI Meeting Corpus (IHM and SDM configurations).

## Tech Stack
Python, PyTorch, HuggingFace Transformers, pyannote.audio, SpeechBrain, HuggingFace Datasets, pyannote.metrics, jiwer, scipy

## Installation
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dataset
AMI Meeting Corpus via HuggingFace `datasets` (`edinburghcstr/ami` or `diarizers-community/ami`).

## License
MIT
