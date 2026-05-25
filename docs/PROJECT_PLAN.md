# Project Plan: Automatic Transcription with Speaker Identification

> **Goal**: Generate meeting audio transcriptions with timestamps and speaker identities, while systematically comparing three diarization modes on the AMI Meeting Corpus dataset (IHM and SDM configurations).
> 
> **Tech Stack**: Python, PyTorch, HuggingFace Transformers, pyannote.audio, SpeechBrain, HuggingFace Datasets, pyannote.metrics, jiwer, scipy

---

## Epic 1: Infrastructure and Project Setup

**Goal**: Prepare the development environment, dependency management, and basic project structure.

### Tasks:

- [ ] **1.1 Repository Initialization**
  - Create Git repository
  - Define `.gitignore` rules (models, datasets, results)
  - Initialize `README.md` with basic project information

- [ ] **1.2 Dependency Management**
  - Create a single `requirements.txt` with all required libraries (use `uv` for environment management):
    - `torch`, `torchaudio`
    - `transformers` (HuggingFace)
    - `pyannote.audio`
    - `speechbrain`
    - `datasets` (HuggingFace)
    - `pyannote.metrics`
    - `jiwer`
    - `scipy`
    - `numpy`, `pandas`, `matplotlib`, `seaborn` (for analysis)
    - `tqdm`, `python-dotenv`
    - `tensorboard` (optional, for lightweight experiment tracking)
  - Version-pin key libraries to ensure reproducibility
  - Set up environment via: `uv venv .venv && uv pip install -r requirements.txt`

- [ ] **1.3 Directory Structure**
  ```
  project/
  ├── data/                    # AMI corpus (raw and preprocessed)
  │   ├── raw/
  │   ├── processed/
  │   └── references/
  ├── models/                  # Downloaded models (Whisper, ECAPA-TDNN)
  ├── src/                     # Source code
  │   ├── asr/
  │   ├── diarization/
  │   ├── integration/
  │   ├── evaluation/
  │   └── utils/
  ├── experiments/             # Experiment configurations
  ├── results/                 # Evaluation results
  │   ├── transcripts/
  │   ├── metrics/
  │   └── visualizations/
  ├── notebooks/               # Jupyter notebooks for analysis
  ├── tests/                   # Targeted unit tests + validation scripts
  │   ├── unit/                # Tests for custom algorithmic logic (mapping, alignment, similarity)
  │   └── validation/          # Smoke tests and integration sanity checks
  └── docs/                    # Documentation
  ```

- [ ] **1.4 GPU Environment Setup**
  - Verify CUDA availability and version
  - Configure PyTorch for GPU acceleration
  - Define hardware config files (batch size, num_workers, etc.)

- [ ] **1.5 Logging and Experiment Tracking**
  - Implement simple file-based logging (stdout + `logs/` directory with experiment ID subfolders)
  - Save experiment configurations and results as JSON/JSONL for traceability
  - **Optional lightweight experiment tracking**: TensorBoard is straightforward (~20 lines: `from torch.utils.tensorboard import SummaryWriter; writer.add_scalar("DER", der)`). Only add if zero friction; file-based logs are the primary source of truth

---

## Epic 2: Data Collection and Preparation

**Goal**: Obtain the AMI Meeting Corpus, perform exploratory analysis, and prepare data for processing.

### Tasks:

- [ ] **2.1 Download AMI Meeting Corpus**
  - Download via HuggingFace `datasets` library (`edinburghcstr/ami` or `diarizers-community/ami`)
  - Verify integrity of downloaded data
  - Organize into `data/raw/` by configuration (IHM, SDM)

- [ ] **2.2 Exploratory Data Analysis (EDA)**
  - Analyze recording durations per meeting
  - Check speaker class distribution
  - Compare IHM vs SDM characteristics (quality, noise, overlap)
  - Visualize audio signals and spectrograms
  - Analyze ground truth transcriptions and speaker labels

- [ ] **2.3 Train/Validation/Test Split Preparation**
  - Implement official AMI split (according to official split)
  - Separate training set (for fine-tuning, if needed) and test set
  - Document split and save metadata

- [ ] **2.4 Audio Preprocessing**
  - Load audio on-the-fly with `torchaudio.load(target_sr=16000)` (AMI is already 16kHz WAV — avoid duplicating ~10GB+ of preprocessed files)
  - Convert to mono channel on-the-fly if needed
  - Process one meeting at a time to manage memory; never load the full 100h corpus into RAM
  - *No need to save preprocessed audio files — load and resample dynamically*

- [ ] **2.5 Ground Truth Annotation Preparation**
  - Parse original XML/RTTM annotations into a uniform format (JSON)
  - Extract word-level transcriptions with timestamps
  - Extract speaker segments (who, from when, to when)
  - Verify annotation consistency

- [ ] **2.6 Reference Sample Preparation (Enrollment Period)**
  - For Scenario 3: extract first ~30 seconds of speech for each speaker from the training/validation set
  - **Filter to speech-only regions using VAD** — do not include silence or overlap in reference embeddings (critical for clean reference quality)
  - Generate reference embeddings (averaged across clean speech clips) for each speaker
  - Save reference embeddings (small .pt files) for reuse across experiments

---

## Epic 3: ASR Pipeline (Automatic Speech Recognition)

**Goal**: Implement the transcription component with word-level timestamps.

### Tasks:

- [ ] **3.1 Whisper Model Integration**
  - Download Whisper model (`openai/whisper-large-v2`) via HuggingFace Transformers
  - Implement inference pipeline for transcription
  - Configure model for English language (AMI is in English)

- [ ] **3.2 VAD-Based Audio Segmentation (Optional / Shared)**
  - Use the same **pyannote.audio VAD model** already loaded in Epic 4.1 — do not implement a second VAD pipeline
  - If Whisper struggles with very long meetings (>30 min), use detected speech regions to chunk audio at silence boundaries; otherwise pass full audio directly to Whisper
  - *Goal: avoid duplicate VAD logic; reuse the pyannote segmentation component*

- [ ] **3.3 Whisper Transcription with Timestamps**
  - Implement batched inference pipeline
  - Generate transcription with utterance-level timestamps
  - Save intermediate results (raw Whisper output)

- [ ] **3.4 Word-Level Timestamp Refinement**
  - **Primary approach**: Use Whisper's built-in word-level timestamps (`return_timestamps="word"` in recent HuggingFace Transformers implementations) if available
  - **Fallback approach**: Distribute word timestamps proportionally within each utterance boundary ( Whisper gives start/end of utterance; split words evenly by character count or token count)
  - *Avoid implementing custom forced phoneme alignment (e.g., wav2vec2.0-based) — it is a mini-project in itself and unnecessary for meeting-level timestamps*

- [ ] **3.5 ASR Pipeline Efficiency**
  - Tune Whisper batch size / chunk length to fit GPU memory without OOM crashes
  - Save raw Whisper outputs (transcriptions + utterance timestamps) to disk so they can be reused across diarization experiments without re-running ASR
  - *Skip Real-Time Factor (RTF) benchmarking — speed is not a research objective of this course project*

- [ ] **3.6 ASR Component Evaluation**
  - Implement WER metric using `jiwer`
  - Evaluate on AMI corpus test set (separately for IHM and SDM)
  - Analyze WER by conditions (clean vs noisy, distant mic, overlap)
  - Save WER results for further analysis

---

## Epic 4: Speaker Diarization

**Goal**: Implement three different speaker identification/diarization modes.

### Tasks:

- [ ] **4.1 pyannote.audio VAD and Segmentation Integration**
  - Download pyannote.audio VAD model
  - Detect speech regions in audio recordings
  - Split segments into ~1.5s clips for embedding extraction

- [ ] **4.2 Speaker Embedding Extraction**
  - Download ECAPA-TDNN model via SpeechBrain (`speechbrain/spkrec-ecapa-voxceleb`)
  - Implement pipeline for extracting 512-dimensional embeddings for each segment
  - Normalize embeddings (L2 normalization)
  - Save embeddings for faster subsequent access

- [ ] **4.3 Scenario 1: Unknown Number of Speakers (AHC auto-k)**
  - Implement Agglomerative Hierarchical Clustering (AHC) with automatic cluster count determination via threshold on linkage distance
  - **Threshold selection**: Use a small validation subset (3–5 meetings) to sweep candidate thresholds and pick the one that minimizes DER before running on the full test set — avoid ad-hoc guessing
  - Post-hoc mapping of clusters to reference speaker IDs using optimal assignment (`scipy.optimize.linear_sum_assignment`)
  - Implement on IHM and SDM configurations

- [ ] **4.4 Scenario 2: Known Number of Speakers (AHC k=4)**
  - Implement AHC with fixed number of clusters k=4
  - Use `average` linkage (standard for speaker diarization) — no need to compare linkage methods; this is not the research question
  - Map clusters to reference IDs using `linear_sum_assignment`
  - Implement on IHM and SDM configurations

- [ ] **4.5 Scenario 3: Reference Identification (Direct Classification)**
  - Generate reference embeddings from enrollment period (first 30s of each speaker)
  - Implement classification based on cosine similarity
  - For each segment: compute similarity with all reference speakers
  - Assign identity with highest similarity (argmax)
  - Tune threshold for rejecting uncertain segments (optional)
  - Implement on IHM and SDM configurations

- [ ] **4.6 Diarization Intermediate Results Tracking and Caching**
  - Save diarization in RTTM format for each scenario
  - Implement utility functions for diarization visualization (plot timeline with speakers)

---

## Epic 5: ASR and Diarization Integration

**Goal**: Merge transcription and speaker identification into a unified output.

### Tasks:

- [ ] **5.1 Timestamp Alignment**
  - Implement deterministic speaker assignment: for each word (with start/end from Whisper), compute overlap duration with every diarization segment; assign the speaker whose segment has the **maximum temporal overlap**
  - If overlap is exactly equal or word falls in a gap, assign speaker of the nearest segment (by start time)
  - *Keep the heuristic simple and deterministic — complex probabilistic alignment is overkill for course-project accuracy goals*

- [ ] **5.2 Final Output Generation**
  - Implement formatter for textual output:
    ```
    [00:00:12 - 00:00:15] Speaker_A: I think we need to change the approach.
    [00:00:15 - 00:00:20] Speaker_B: I agree, but first we must analyze the data.
    ```
  - Implement formatter for JSON output (structure suitable for evaluation and further processing)
  - Include metadata (recording name, scenario, microphone configuration)

- [ ] **5.3 Refinement Heuristics**
  - Merge adjacent diarization segments belonging to the same speaker if separated by <0.5s (reduces fragmentation in textual output)
  - Apply minimum segment duration filter (e.g., discard segments <0.2s) to suppress spurious VAD noise detections
  - *Do not implement custom overlap resolution logic — evaluation already skips overlapping regions (`skip_overlap=True`), and pyannote handles overlap detection upstream*

---

## Epic 6: Evaluation and Metrics

**Goal**: Implement all evaluation metrics and run evaluation for all scenarios.

### Tasks:

- [ ] **6.1 DER (Diarization Error Rate) Implementation**
  - Integrate `pyannote.metrics` for DER calculation
  - Configure parameters: collar=0.25s, skip_overlap=True
  - Calculate DER components: Missed Speech, False Alarm, Speaker Confusion

- [ ] **6.2 JER (Jaccard Error Rate) Implementation**
  - Integrate `pyannote.metrics` for JER calculation
  - Compare with DER (analyze per-speaker weighting)

- [ ] **6.3 Speaker Identification Accuracy Implementation**
  - For Scenario 3: calculate percentage of correctly identified segments
  - Generate confusion matrices per speaker

- [ ] **6.4 WER (Word Error Rate) Implementation**
  - Integrate `jiwer` library
  - Evaluate impact of diarization on WER (whisper-only vs integrated)

- [ ] **6.5 Config-Driven Experiment Runner**
  - Define 6 experiment configs as JSON/Python dicts (scenario ID, mic type, model paths, threshold values)
  - Write one `run_experiment.py` that loads a config and executes the full pipeline end-to-end
  - Create a top-level `run_all_experiments.sh` (or Python loop) that iterates over the 6 configs, aggregates results, and saves per-recording metrics — eliminates manual parameter passing errors

- [ ] **6.6 Results Analysis**
  - Compare DER gap between IHM and SDM per scenario
  - Verify hypotheses:
    - Scenario 3 smallest DER gap IHM vs SDM
    - Scenario 1 largest DER gap IHM vs SDM
    - Scenario 2 between these two
  - Identify error cases (worst recordings, root cause analysis)

---

## Epic 7: Visualization and Results Presentation

**Goal**: Visually present results and enable performance interpretation.

### Tasks:

- [ ] **7.1 Diarization Visualization**
  - Plot timeline graphs: ground truth vs predicted diarization
  - Color segments by speaker
  - Visualize overlapping regions

- [ ] **7.2 Metrics Visualization**
  - Bar charts: DER by scenario and microphone
  - Grouped bar chart: IHM vs SDM comparison
  - Box plots: DER/WER distribution across recordings
  - Confusion matrix for Scenario 3

- [ ] **7.3 Transcription Examples**
  - Manual selection of representative segments (successful and failure cases)
  - Generate side-by-side comparisons: ground truth vs predicted
  - Save in format suitable for paper/presentation display

- [ ] **7.4 Jupyter Notebook for Analysis**
  - Interactive notebook for exploratory analysis of results
  - Report descriptive statistics (mean, std, min, max) per scenario and microphone configuration
  - *Avoid t-tests/ANOVA — with only 2 microphone configurations and limited test-set meetings, n is too small for parametric statistics to be meaningful or honest. Descriptive tables + clear visualizations are stronger*

---

## Epic 8: Documentation and Reproducibility

**Goal**: Ensure the entire experiment is reproducible and well-documented.

### Tasks:

- [ ] **8.1 Pipeline Documentation**
  - Document each pipeline phase (ASR, VAD, embedding, diarization, integration) with a simple block diagram in the final report
  - Describe hyperparameters and their chosen values
  - *No need for a formal Data Flow Diagram — a simple boxes-and-arrows architecture diagram is sufficient*

- [ ] **8.2 Experiment Documentation**
  - Describe all 6 experiments (scenario × microphone)
  - Configuration files (YAML/JSON) for each experiment
  - Experiment log (what was tuned, changes, decisions)

- [ ] **8.3 Reproducibility**
  - Implement seed fixing (random, numpy, torch)
  - Save versions of all libraries (`pip freeze > requirements.txt` or `conda env export > environment.yml`)
  - Record exact Python, PyTorch, and CUDA versions in README
  - *Skip Docker — containerizing is unnecessary for a personal course project. A frozen requirements file is sufficient*

- [ ] **8.4 Testing and Validation**
  - **Targeted unit tests** (~5–7 tests) for custom algorithmic logic that validation cannot reliably exercise:
    - Cluster-to-speaker mapping via `linear_sum_assignment` (correctness of cost matrix orientation)
    - Cosine similarity normalization and top-k classification (verify against hand-computed examples)
    - Timestamp alignment edge cases (boundary words, gaps, overlaps)
  - **Validation scripts** (smoke/integration checks) run in seconds:
    - `validate_pipeline.py` — run on one short meeting; verify outputs are well-formed, all 4 speakers appear, no NaNs in embeddings
    - `validate_metrics.py` — compute DER on a tiny subset with known answer; assert within expected bounds
    - `validate_alignment.py` — check that word timestamps and speaker segments overlap logically
  - Keep unit tests lightweight; avoid mocking PyTorch/HuggingFace pipelines

- [ ] **8.5 Technical Report Writing**
  - Structure: Introduction, Methodology, Experimental Results, Discussion, Conclusion
  - Include tables with metrics and charts
  - Compare with expected hypotheses

---

## Dependency Graph and Execution Order

```
Epic 1 (Infrastructure)
    │
    ▼
Epic 2 (Data) ────────────────┐
    │                           │
    ▼                           │
Epic 3 (ASR) ◄──────────────────┤
    │                           │
    ▼                           │
Epic 4 (Diarization) ◄──────────┘
    │
    ▼
Epic 5 (Integration)
    │
    ▼
Epic 6 (Evaluation) ◄───────────┐
    │                           │
    ▼                           │
Epic 7 (Visualization) ◄────────┘
    │
    ▼
Epic 8 (Documentation)
```

**Recommended Implementation Order**:
1. Set up infrastructure (Epic 1)
2. Obtain and understand data (Epic 2)
3. Implement ASR component (Epic 3)
4. Implement diarization (Epic 4) — can be done in parallel with ASR once data is ready
5. Integrate ASR + diarization (Epic 5)
6. Evaluate and analyze (Epic 6)
7. Visualize (Epic 7)
8. Document (Epic 8) — in parallel throughout the project

---

## Key Decision Points

1. **Whisper Model Choice**: `whisper-large-v2` or smaller model for faster iteration?
   *Recommendation*: Start with `whisper-base` or `whisper-small` for faster iteration, switch to `whisper-large-v2` for final results.

2. **Forced Alignment**: Implement custom alignment or use WhisperX approach?
   *Recommendation*: First try `aeneas` or `torchaudio` forced alignment; if insufficiently accurate, consider WhisperX approach.

3. **AHC Parameters**: Which linkage and threshold parameters for Scenario 1?
   *Recommendation*: Experiment with `average` linkage and threshold of 0.5 on training/validation set.

4. **Reference Enrollment**: Is 30 seconds sufficient for quality references?
   *Recommendation*: If results are poor, consider longer enrollment or selection of cleanest segments.

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Domain mismatch (VoxCeleb → AMI SDM) | High | High | Test out-of-the-box first; consider fine-tuning ECAPA-TDNN on AMI if needed |
| Overlapping speech degrades performance | High | Medium | Explicitly mark overlap regions; evaluate with `skip_overlap=True` |
| Whisper timestamps not precise enough | Medium | Medium | Use forced alignment for word-level timestamps |
| pyannote.audio dependencies problematic | Medium | Medium | Use precompiled wheels or Docker; fallback to alternative VAD models |
| GPU memory insufficient for batch processing | Medium | Medium | Reduce batch size, use mixed precision, process one meeting at a time, implement CPU fallback for OOM |
| Naive data loading (full corpus in RAM) | Medium | High | Stream one meeting at a time; do not load entire AMI corpus into memory |
| Corrupt/missing audio files in download | Low | Medium | Add file-existence checks at start of pipeline; skip and log corrupted files rather than crashing |

---

*This plan is a living document and may be updated as the project progresses and new information is discovered.*
