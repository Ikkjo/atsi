# Agent Handoff Document

> **Project**: Automatic Transcription with Speaker Identification  
> **GitHub Repo Name**: `automatic-transcription-speaker-id`  
> **Deadline**: June 5, 2026 (implementation freeze) / June 19, 2026 (report submission)  
> **Current Date**: May 25, 2026  
> **Status**: Planning complete, ready for implementation  

---

## 1. Project Overview

You are implementing a course project that automatically transcribes meeting audio and identifies which speaker spoke when. The project systematically compares **three speaker diarization modes** on the **AMI Meeting Corpus**:

1. **Scenario 1**: Unknown number of speakers — Agglomerative Hierarchical Clustering (AHC) with automatic cluster count determination
2. **Scenario 2**: Known number of speakers (k=4) — AHC with fixed clusters
3. **Scenario 3**: Reference-based identification — Direct classification using cosine similarity with pre-extracted speaker reference embeddings

All scenarios run on both **IHM** (Individual Headset Microphones) and **SDM** (Single Distant Microphone) configurations, yielding 6 total experiments.

### Expected Output
```
[00:00:12 - 00:00:15] Speaker_A: I think we need to change the approach.
[00:00:15 - 00:00:20] Speaker_B: I agree, but first we must analyze the data.
```

And a JSON format for evaluation.

### Key Metrics
- **DER** (Diarization Error Rate) — primary metric
- **WER** (Word Error Rate) — secondary metric for ASR quality
- **JER** (Jaccard Error Rate) — alternative diarization metric
- **Speaker Identification Accuracy** — for Scenario 3

---

## 2. What Has Been Done (By Previous Agent)

- ✅ Extracted and translated the original project proposal from PDF
- ✅ Created detailed project plan (`docs/PROJECT_PLAN.md`) with 8 Epics and ~40 tasks
- ✅ Created realistic timeline (`docs/PROJECT_TIMELINE.md`) accounting for ~1-2 hours/day constraint
- ✅ Established project directory structure
- ✅ Created `.gitignore`, `README.md`, `requirements.txt`
- ✅ Defined scope reductions (see below)

**Nothing else is implemented yet.** No code exists in `src/`. No data has been downloaded.

---

## 3. Architecture Decisions Already Made (Do Not Change Without Discussion)

### Tech Stack
- **Python** + **uv** for environment management (`uv venv .venv && uv pip install -r requirements.txt`)
- **PyTorch** + **HuggingFace Transformers** for Whisper ASR
- **pyannote.audio** for VAD and segmentation
- **SpeechBrain** for ECAPA-TDNN speaker embeddings
- **pyannote.metrics** for DER/JER evaluation
- **jiwer** for WER evaluation
- **scipy** for `linear_sum_assignment` (cluster mapping)

### Simplifications (Agreed Upon)
1. **No word-level timestamps** — Whisper utterance-level timestamps are sufficient. Assign entire utterance to dominant speaker by temporal overlap.
2. **No forced phoneme alignment** — Whisper native timestamps or proportional distribution within utterance boundaries.
3. **No textual transcript formatting** — JSON output with segments is sufficient for evaluation and report.
4. **No EDA notebooks** — Skip exploratory data analysis plots.
5. **No Docker** — `requirements.txt` + `uv` is sufficient.
6. **No W&B/TensorBoard** — File-based JSONL logging only.
7. **No formal unit tests** — One `smoke_test.py` plus 5-7 targeted unit tests for custom algorithms.
8. **No statistical tests** (t-tests/ANOVA) — Descriptive statistics only (small n).

### Key Design Patterns
- **Config-driven experiments**: Each of the 6 experiments defined as a JSON/Python config dict
- **One experiment runner script**: `run_experiment.py` loads config and executes full pipeline
- **Stream one meeting at a time**: Never load full AMI corpus into RAM
- **Save intermediate outputs**: Whisper transcripts and embeddings saved to disk for reuse
- **Process one meeting at a time**: Manage memory, avoid OOM

---

## 4. Where to Start (Implementation Order)

### Phase 1: Infrastructure & Data (Epics 1-2)
**Start here.** These must be solid before anything else.

1. **Environment setup**: Install dependencies via `uv`, verify GPU/CUDA works
2. **AMI data loading**: Use HuggingFace `datasets` to load `edinburghcstr/ami` or `diarizers-community/ami`
3. **Ground truth parsing**: Extract word-level transcripts and speaker segments from the dataset's built-in annotations
4. **Train/validation/test split**: Use the official AMI split (provided by the dataset)
5. **Reference embeddings (Scenario 3)**: 
   - Extract first ~30 seconds of speech per speaker from training/validation meetings
   - **Filter to speech-only using VAD** (critical — do not include silence in reference embeddings)
   - Extract ECAPA embeddings, average them, save as `.pt` files

### Phase 2: Core Pipeline (Epics 3-4)
**ASR and Diarization can be developed independently once data is loaded.**

**ASR Track**:
- Integrate Whisper via HuggingFace (`openai/whisper-base` for development speed, `large-v2` for final results)
- Produce utterance-level JSON: `{"start": float, "end": float, "text": str}`
- Save to `results/whisper_transcripts/`

**Diarization Track**:
- Integrate pyannote.audio segmentation model
- Integrate SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- For each detected segment, extract 512-dim embedding, L2-normalize
- **Scenario 2** (easiest): `sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='average')`
- **Scenario 3**: Cosine similarity to 4 reference embeddings, argmax
- **Scenario 1**: Same clustering but `distance_threshold=X`. Try X ∈ [0.3, 0.5, 0.7], pick whichever gives ~4 clusters on a validation meeting. Document the chosen X as a limitation.

### Phase 3: Integration (Epic 5)
- For each Whisper utterance, find diarization segments that overlap
- Assign utterance to speaker with **maximum temporal overlap**
- Save JSON output: `{"start": float, "end": float, "speaker": str, "text": str}`
- Write `smoke_test.py`: run Scenario 2 on 30-second clip, assert output has utterances and 4 unique speakers

### Phase 4: Evaluation (Epic 6)
- Implement config-driven `run_experiment.py` that loads experiment configs and runs full pipeline
- Compute DER via `pyannote.metrics` (collar=0.25s, skip_overlap=True)
- Compute WER via `jiwer`
- Run all 6 experiments (or minimum 4 if time-constrained: Scenario 2+3 on both mics)
- Save `results/metrics/all_experiments.json`

### Phase 5: Visualization (Epic 7)
- Figure 1: Bar chart — DER by scenario × microphone (grouped bars)
- Figure 2: Box plot — DER distribution across test meetings per scenario
- Figure 3: Confusion matrix for Scenario 3 (only if Scenario 3 completed)
- Save to `results/visualizations/`

### Phase 6: Documentation & Freeze (Epic 8)
- `uv pip freeze > requirements.txt`
- Seed fixing verified
- README updated with install + run instructions
- Git tag `v1.0-frozen`
- Copy results/visualizations to `report_materials/`

---

## 5. Critical Technical Details

### AMI Dataset
- **Source**: HuggingFace `datasets` — `edinburghcstr/ami` or `diarizers-community/ami`
- **Content**: ~100 hours, 4 speakers per meeting, both IHM and SDM recordings
- **Sampling rate**: 16kHz WAV (already correct)
- **Ground truth**: Word-level transcripts + speaker labels provided by dataset
- **Target**: Classification into 4 speaker classes per meeting

### Evaluation Protocol
- Use official AMI train/test split
- Evaluate on test set only
- For Scenario 3, extract reference samples from **training/validation** set (first 30s per speaker, VAD-filtered)
- **DER formula**: (Missed Speech + False Alarm + Speaker Confusion) / Total Speech Duration
- **Collar**: 0.25s tolerance around segment boundaries
- **Skip overlap**: True (exclude overlapping regions from DER calculation)

### Cluster-to-Speaker Mapping
- After clustering (Scenarios 1 and 2), clusters are unlabeled (0, 1, 2, 3...)
- Use `scipy.optimize.linear_sum_assignment` to find optimal mapping to ground truth speaker IDs
- This minimizes speaker confusion in DER calculation

### Memory Management
- **Never load full corpus into RAM**
- Process **one meeting at a time**
- Audio loaded via `torchaudio.load(target_sr=16000)` on-the-fly
- Save embeddings and transcripts to disk, load per-meeting
- GPU batch sizes: tune to avoid OOM (start small: batch_size=1-2 for Whisper)

### Hypotheses to Verify
1. Scenario 3 shows the **smallest** DER gap between IHM and SDM
2. Scenario 1 shows the **largest** DER gap between IHM and SDM
3. Scenario 2 is **between** the two

These hypotheses must be checked and reported honestly, even if negative.

---

## 6. Project Structure

```
automatic-transcription-speaker-id/
├── data/
│   ├── raw/                    # AMI corpus (not in git)
│   ├── processed/              # Intermediate outputs (not in git)
│   └── references/             # Reference embeddings for Scenario 3
├── models/                     # Downloaded models (not in git)
├── src/
│   ├── asr/                    # Whisper integration
│   ├── diarization/            # pyannote + ECAPA + clustering
│   ├── integration/            # ASR + diarization alignment
│   ├── evaluation/             # DER, WER, metrics
│   └── utils/                  # Shared utilities
├── experiments/                # Config files for 6 experiments
├── results/
│   ├── transcripts/            # Final outputs
│   ├── metrics/                # DER, WER per experiment
│   └── visualizations/         # Figures
├── notebooks/                  # (optional analysis notebooks)
├── tests/
│   ├── unit/                   # ~5-7 targeted tests for custom logic
│   └── validation/             # smoke_test.py + integration checks
├── docs/
│   ├── PROJECT_PLAN.md         # Full project plan (8 Epics)
│   ├── PROJECT_TIMELINE.md     # Day-by-day schedule
│   └── project_proposal.txt    # Original proposal (translated)
├── logs/                       # Experiment logs (not in git)
├── report_materials/           # Copy of final results for report
├── .gitignore
├── README.md
├── requirements.txt
└── HANDOFF.md                  # This file
```

---

## 7. Common Pitfalls to Avoid

1. **Domain mismatch**: ECAPA-TDNN trained on VoxCeleb (YouTube interviews), applied to AMI meetings (distant mics, reverberation). Expect degradation on SDM. This is a valid finding.
2. **Reference embedding quality**: If Scenario 3 underperforms, first check if reference embeddings include silence. They MUST be VAD-filtered.
3. **pyannote.audio dependency issues**: If installation fails, try pre-built wheels from pyannote GitHub releases. If still stuck, use `torchaudio` VAD as temporary fallback.
4. **OOM crashes**: Start with `whisper-base` and batch_size=1. Scale up only after verifying stability.
5. **Whisper language**: AMI is English. Ensure Whisper model configured for English.
6. **Ground truth alignment**: When mapping clusters to speakers with `linear_sum_assignment`, verify cost matrix orientation. A common bug is transposing the matrix, which silently mislabels all speakers.
7. **Timestamp alignment**: The alignment between Whisper utterances and diarization segments is segment-level (not word-level). Assign the entire utterance to the speaker with maximum overlap. Do not try to split words across speakers.

---

## 8. If Things Go Wrong (Contingency Plan)

| Problem | Solution |
|---------|----------|
| pyannote.audio won't install | Use pre-built wheels; fallback to `torchaudio` VAD + manual segmentation |
| GPU unavailable | Use CPU for development; GPU only needed for batch experiment execution |
| AMI download fails | Use direct download from AMI official website; write manual loader |
| Scenario 3 worse than expected | **Valid negative result** — document honestly. Check reference quality, enrollment period length |
| One experiment crashes repeatedly | Skip it, report remaining 5. Partial results > no results |
| Running out of time | **Priority**: Complete Scenario 2 + 3 on IHM (minimum viable: 2 scenarios × 1 mic = 2 experiments) |
| Report reveals methodological flaw | Document as limitation in Discussion section. Do not hide it. |

---

## 9. Files You Should Read First

1. **`docs/PROJECT_PLAN.md`** — Full project plan with all 8 Epics and detailed tasks
2. **`docs/PROJECT_TIMELINE.md`** — Day-by-day schedule with deliverables
3. **`docs/project_proposal.txt`** — Original project proposal (translated from PDF)
4. **`README.md`** — Basic project info and install instructions
5. **`HANDOFF.md`** — This file (reference for context)

---

## 10. Success Criteria

The project is successful if:
- [ ] At least **4 experiments** complete with DER computed (preferably all 6)
- [ ] DER results are computed correctly using `pyannote.metrics`
- [ ] Results JSON exists with DER/WER per experiment
- [ ] 2-3 visualization figures exist showing scenario comparisons
- [ ] Code is reproducible from `requirements.txt` + README instructions
- [ ] Git history shows incremental commits
- [ ] Report can be written using only materials in `report_materials/`

**The project is acceptable (passing grade) if:**
- [ ] Scenario 2 + Scenario 3 complete on at least one microphone config
- [ ] DER computed and compared between the two scenarios
- [ ] At least one figure exists
- [ ] Code runs end-to-end on at least one test meeting

---

*This handoff document was prepared on May 25, 2026. The next agent should begin with Epic 1 (Infrastructure) and proceed through Epic 8 by June 5, 2026.*
