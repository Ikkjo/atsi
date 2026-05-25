# Revised Project Timeline — ~1–2 Hours/Day Constraint

> **Hard Constraint**: ~1–2 hours per day maximum (including weekends)  
> **Total Implementation Budget**: ~20 hours (May 25 – June 5)  
> **Implementation Deadline**: June 5, 2026  
> **Report Submission**: June 19, 2026  
> **Today**: May 25, 2026 (Monday)

**This is a tight but feasible schedule if and only if scope is aggressively reduced.** The original plan assumed ~60 hours of implementation. This version assumes ~20.

---

## Calendar Overview

| Week | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|------|-----|-----|-----|-----|-----|-----|-----|
| **May 25–31** | **25** 🔧 | **26** 📦 | **27** 🎙️ | **28** 👥 | **29** 🔗 | **30** ⚡ | **31** ⚡ |
| **Jun 1–5** | 📊 | 📊 | 🧊 | 🧊 | **5** 🛑 | — | — |
| **Jun 6–19** | 📝 | 📝 | 📝 | 📝 | 📝 | 📝 | 📝 |

**Legend**: 🔧 Setup | 📦 Data | 🎙️ ASR | 👥 Diarization | 🔗 Integration | ⚡ Experiments | 📊 Analysis | 🧊 Freeze/Buffer | 📝 Report | 🛑 Hard Stop

---

## Philosophy: Ruthless Scope Reduction

With ~20 hours total, **every hour must directly serve the final report**. If a task does not produce a result you will reference in the report, it is cut.

### What is CUT entirely (do not touch)

| Cut Item | Why |
|----------|-----|
| Word-level timestamps + alignment | Whisper utterance-level timestamps are sufficient. Segment-level speaker assignment is acceptable for a course project. |
| Forced alignment (wav2vec2 / WhisperX-style) | Mini-project in itself; ~4–6 hours minimum. Not needed for DER evaluation. |
| Textual transcript output (`[00:00:12] Speaker_A: ...`) | JSON output with segments is sufficient for evaluation and the report. Textual formatting is polish, not science. |
| EDA notebook | No grade depends on spectrograms. Know the dataset from the paper, not from plotting. |
| Jupyter notebook for analysis | Static matplotlib scripts are faster and reproducible. |
| Formal unit tests | One `smoke_test.py` that runs the pipeline on 30 seconds of audio is enough. |
| Docker / conda env export | `requirements.txt` + `uv` is sufficient. |
| Real-Time Factor (RTF) benchmarking | Not a research objective. |
| Overlap resolution heuristics | Evaluation skips overlap (`skip_overlap=True`). No need to resolve it in output. |
| W&B / TensorBoard | File-based JSON logs are enough for 6 experiments. |
| Data Flow Diagram | A simple architecture paragraph in the report replaces a diagram. |
| Per-recording WER breakdown by condition | Compute overall WER per scenario; skip clean/noisy/overlap sub-analysis. |

### What is SIMPLIFIED dramatically

| Original Plan | Simplified To |
|---------------|---------------|
| Parse RTTM/XML into custom JSON | Use HuggingFace `datasets` AMI loader directly; work with whatever structure it provides |
| 3 validation scripts | One `smoke_test.py` that runs end-to-end on a 30-second clip and asserts output exists |
| 6–8 publication-quality figures | 3 figures: (1) DER bar chart by scenario×mic, (2) box plot of DER distribution, (3) confusion matrix for Scenario 3 |
| Word-level alignment heuristic | Assign entire Whisper **utterance** to the dominant speaker in that time window (segment-level) |
| Sweep threshold on validation set for Scenario 1 | Try 2–3 threshold values by hand on one meeting, pick the best, document it as a limitation |
| Extensive README | 10-line README: install command, one-liner to run experiments, where results live |

---

## Detailed Day-by-Day Plan

### 🔧 May 25 (Monday) — Setup [~1.5h]
**Goal**: Repo exists, libraries install, GPU works.

- [ ] `git init`, `.gitignore` (models/, data/, results/), basic directory structure
- [ ] `requirements.txt` with pinned versions, `uv venv .venv && uv pip install -r requirements.txt`
- [ ] One script `test_imports.py` that imports torch, transformers, pyannote, speechbrain and prints `torch.cuda.is_available()`

**If stuck >30 min on any dependency**: Skip it, use CPU for now, move on. Fix on May 30 if time permits.

**Deliverable**: `python test_imports.py` prints `True` (or `False` for CPU) without errors.

---

### 📦 May 26 (Tuesday) — Data Loading [~1.5h]
**Goal**: Load AMI via HuggingFace, extract ground truth, prepare references.

- [ ] Load `edinburghcstr/ami` or `diarizers-community/ami` via `datasets` library
- [ ] Identify train/test split fields; save a simple JSON with meeting IDs per split
- [ ] Extract word-level ground truth annotations (the HuggingFace dataset may already provide this)
- [ ] For Scenario 3: extract first ~30s of speech per speaker from **training** meetings; use pyannote VAD to find actual speech regions (skip silence); average their embeddings

**Critical**: Do NOT download the full ~10GB corpus manually. The HuggingFace loader streams what you need.

**Deliverable**: A Python script `load_data.py` that returns `train_ids`, `test_ids`, and `reference_embeddings.pt`.

---

### 🎙️ May 27 (Wednesday) — ASR Pipeline [~2h]
**Goal**: Whisper produces utterance-level transcripts with timestamps.

- [ ] Load Whisper (`openai/whisper-base` or `small` for speed; use `large-v2` only if time remains on May 30) via HuggingFace `pipeline`
- [ ] Run on one test meeting, save JSON: list of utterances with `{"start": float, "end": float, "text": str}`
- [ ] **No word-level timestamps.** Segment-level is acceptable.
- [ ] Save outputs to `results/whisper_transcripts/` so you never re-run ASR

**Deliverable**: `results/whisper_transcripts/ES2001a.json` exists and contains segments with text and timestamps.

---

### 👥 May 28 (Thursday) — Diarization Core [~2h]
**Goal**: pyannote segmentation + ECAPA embeddings work.

- [ ] Load pyannote.audio segmentation model (`pyannote/segmentation`)
- [ ] Load SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- [ ] Run on one meeting: get speech segments (~1.5s each), extract 512-dim embeddings, L2-normalize
- [ ] Save embeddings to `results/embeddings/` (small `.pt` files)

**Do NOT implement all 3 scenarios today.** Just get the building blocks working.

**Deliverable**: `results/embeddings/ES2001a.pt` exists with shape `(N_segments, 512)`.

---

### 👥 May 29 (Friday) — Three Scenarios [~2h]
**Goal**: All three scenarios produce speaker labels for one meeting.

- [ ] **Scenario 2** (easiest): `sklearn.cluster.AgglomerativeClustering(n_clusters=4, linkage='average')` on embeddings; map clusters to speakers via `linear_sum_assignment`
- [ ] **Scenario 3**: cosine similarity of each segment embedding to 4 reference embeddings; argmax
- [ ] **Scenario 1** (hardest): Same clustering but without `n_clusters=4`. Try `distance_threshold=X` where X is one of [0.3, 0.5, 0.7]. Pick whichever gives 4 clusters on one validation meeting. Document the chosen X. **Do not sweep extensively.**
- [ ] Save all diarizations as RTTM or simple JSON

**Deliverable**: Three JSON files in `results/diarization/ES2001a_scenario{1,2,3}_IHM.json`.

---

### 🔗 May 30 (Saturday) — Integration + Smoke Test [~2h]
**Goal**: One end-to-end transcript exists. Pipeline is believable.

- [ ] For each utterance from Whisper, find overlapping diarization segments; assign the speaker with **most temporal overlap** (simple heuristic, 10 lines of code)
- [ ] Generate JSON output: list of utterances with `"speaker"` field added
- [ ] Write `smoke_test.py`: runs Scenario 2 on a 30-second clip, asserts output JSON has >0 utterances and 4 unique speakers
- [ ] Run `smoke_test.py`; fix any crash

**Deliverable**: `results/transcripts/ES2001a_scenario2_IHM.json` exists and `python smoke_test.py` passes.

---

### ⚡ May 31 (Sunday) — Batch Experiments Begin [~2h]
**Goal**: Start the long-running jobs.

- [ ] Write a 20-line script `run_all.py` that loops over scenarios [1,2,3] and mics [IHM, SDM], runs the pipeline, saves results
- [ ] Launch `run_all.py` on the **full test set** (this will take hours of GPU time; mostly babysitting)
- [ ] While GPU runs, compute DER for any completed experiments using `pyannote.metrics`

**If one scenario crashes**: Skip it, debug later. Prioritize completing Scenario 2 + 3 first (most interesting comparison). Scenario 1 is a bonus.

**Deliverable**: At least 2–4 experiments complete with DER computed.

---

### ⚡ Jun 1 (Monday) — Experiments Continue + DER [~1.5h]
**Goal**: All experiments that will finish are finished.

- [ ] Check GPU outputs; restart any crashed jobs with simpler settings (smaller batch size, CPU fallback)
- [ ] Compute DER for all finished experiments; save to `results/metrics/der_results.json`
- [ ] Compute WER with `jiwer` (if transcripts are ready)

**Honesty check**: If fewer than 4 experiments completed by tonight, **permanently drop Scenario 1** and focus on completing Scenario 2 + 3 on both mics. Two scenarios compared across two mics is still a valid project.

**Deliverable**: `results/metrics/der_results.json` has DER values for at least 4 experiments.

---

### 📊 Jun 2 (Tuesday) — Figures + Sanity Check [~1.5h]
**Goal**: Report has visual evidence.

- [ ] Figure 1: Bar chart — DER by scenario (x-axis) and mic (grouped bars)
- [ ] Figure 2: Box plot — DER distribution across test meetings (one box per scenario×mic)
- [ ] Figure 3 (only if Scenario 3 done): Confusion matrix for Scenario 3 speaker identification accuracy
- [ ] Verify hypotheses visually: does Scenario 3 have smallest IHM–SDM gap? Is Scenario 1 largest?
- [ ] Write 2–3 sentences of interpretation per figure (save in `results/visualizations/captions.txt`)

**Deliverable**: `results/visualizations/` contains 2–3 PNG files + caption text.

---

### 🧊 Jun 3 (Wednesday) — Buffer / Catch-up [~1h]
**Goal**: Absorb any slippage.

- [ ] If experiments incomplete: this is the last day to run them
- [ ] If experiments complete but DER seems wrong: debug one meeting manually
- [ ] If everything done: generate `requirements.txt` freeze (`uv pip freeze > requirements.txt`)

**Deliverable**: All code results that will exist, exist.

---

### 🧊 Jun 4 (Thursday) — Final Freeze Prep [~1h]
**Goal**: Nothing breaks on Jun 5.

- [ ] Run `smoke_test.py` one final time
- [ ] `git add`, `git commit -m "freeze: all experiments complete"`
- [ ] Copy `results/` and `results/visualizations/` to `report_materials/`
- [ ] Write 5-line README: `uv pip install -r requirements.txt && python run_all.py`

**Deliverable**: Repo is clean. `report_materials/` has everything needed for the report.

---

### 🛑 Jun 5 (Friday) — HARD STOP [~0.5h]
**Goal**: Zero code changes after today.

- [ ] Final `git tag freeze-jun5`
- [ ] No commits to `main` after 12:00 PM
- [ ] If you think of a bug: write it in `BUGS.md`, fix it only if catastrophic

---

## Phase 2: Report Writing (Jun 6 – Jun 19)

You now have **14 days** and no coding pressure. Use this time well.

| Period | Focus | Hours (your own schedule) |
|--------|-------|---------------------------|
| Jun 6–9 | Draft: Intro, Methodology, Experimental Setup | 6–10 |
| Jun 10–13 | Draft: Results, Discussion | 6–10 |
| Jun 14–16 | Revision pass: figures, tables, references | 4–6 |
| Jun 17–18 | Final polish, spell-check, PDF generation | 3–4 |
| **Jun 19** | **SUBMIT** | — |

**Critical**: The report should explicitly mention limitations and honest negative results. If Scenario 3 underperforms, say so and hypothesize why (domain mismatch, enrollment too short). This strengthens the paper, not weakens it.

---

## Priority Order if Things Go Wrong

If you fall behind, drop items in this order:

1. **Drop Scenario 1** (unknown k) — hardest to tune, least reliable. Keep Scenario 2 (known k=4) and Scenario 3 (reference ID). These two already make a strong comparison: clustering vs. direct classification.
2. **Drop IHM or SDM** — if only one mic configuration works, report it and note the other as future work.
3. **Drop WER analysis** — DER is the primary metric. WER is secondary.
4. **Drop Figure 3 (confusion matrix)** — bar chart + box plot are sufficient.
5. **Drop JSON output format** — RTTM or simple CSV with (start, end, speaker, text) is enough.
6. **Drop README polish** — as long as `requirements.txt` exists, reproducibility is credible.

**Minimum viable project** (if everything goes wrong):
- Scenario 2 (k=4) + Scenario 3 (reference) on IHM only
- DER computed for both
- One bar chart
- Report explaining why the comparison matters, even with limited scope

This is still a passing project. A narrow but correct result is better than a broad but broken one.

---

## Time Budget Summary

| Phase | Days | Hours/Day | Total Hours | What You Get |
|-------|------|-----------|-------------|--------------|
| Setup + Data | 2 | 1.5 | 3 | Working environment, loaded AMI data |
| ASR + Diarization Core | 2 | 2.0 | 4 | Whisper transcripts, pyannote + ECAPA working |
| 3 Scenarios | 1 | 2.0 | 2 | All 3 scenarios implemented (even if crude) |
| Integration + Smoke Test | 1 | 2.0 | 2 | End-to-end pipeline runs without crashing |
| Experiments | 2 | 2.0 | 4 | 4–6 experiments complete with DER |
| Analysis + Figures | 1 | 1.5 | 1.5 | 2–3 figures, hypothesis check |
| Buffer + Freeze | 2 | 1.0 | 2 | Catch-up, final `requirements.txt`, tag |
| **Total Implementation** | **11** | **1.6 avg** | **~18.5** | **Core project complete** |
| Report Writing | 14 | (your pace) | — | Full technical report |

**This assumes no major dependency disasters.** If pyannote.audio refuses to install, add 1 day (May 30 buffer absorbs it). If GPU is unavailable, use CPU — slower but functional.

---

*This timeline is pessimistic by design. Most tasks are estimated at 1.5–2 hours, which is tight but forces focus. The June 5 freeze is non-negotiable to protect report writing time.*
