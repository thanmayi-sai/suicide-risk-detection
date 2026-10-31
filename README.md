# Suicide-Risk Detection Using NLP-Based Text and Speech Analysis

Final-year B.Tech Information Technology project.

## Project Team
- S. Thanmayi — 23761A1251
- Ch. Venu Madhav — 24765A1201
- M. Samuel Babu — 23761A1235

**Guide:** Dr. G. Rajendra, Associate Professor  
**Department:** Information Technology, Lakireddy Bali Reddy College of Engineering (LBRCE)

## Overview
This repository implements a research-screening prototype with three modes:

1. **Text only** — enhanced fast word + character TF-IDF model trained on SDCNL labels corrected with a BERT → UMAP → GMM confidence-based label-correction stage.
2. **Speech only** — EATD acoustic feature analysis using Logistic Regression.
3. **Text + Speech** — transparent late-fusion research decision layer.

> This project is for academic/research screening only. It is not a clinical diagnostic system.

## Primary Verified Text Result

The production/demo text model is a **fast word+character TF-IDF + calibrated LinearSVC** trained using SDCNL-style confidence-corrected labels.

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| **85.75%** | **93.98%** | **78.00%** | **85.25%** | **95.04%** |

Label-correction pipeline used before training:

```
BERT mean-pooled embeddings
        ↓
UMAP dimensionality reduction
        ↓
Gaussian Mixture Model
        ↓
confidence threshold τ = 0.90
        ↓
corrected SDCNL labels
        ↓
fast word+char TF-IDF + calibrated LinearSVC
```

The hyperparameters of the final fast classifier were selected using corrected-training **5-fold out-of-fold evaluation**; the test labels were not used to select C or the classifier threshold.

### Separate offline label-correction research experiment

A BERT-embedding Logistic Regression classifier evaluated on the SDCNL-style corrected test labels reached:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 93.14% | 95.31% | 91.50% | 93.37% | 98.53% |

This 93.14% result is retained as an **offline research comparison**, not the fast production model.

<details>
<summary><b>Standard original-web-label baselines (before SDCNL-style label correction)</b></summary>

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 59.37% | 57.77% | 75.13% | 65.32% | 63.58% |
| Frozen BERT + Logistic Regression | 64.12% | 63.77% | 68.39% | 66.00% | 72.27% |
| TF-IDF + Calibrated Linear SVM | 72.56% | 71.92% | 75.65% | 73.74% | 78.22% |

These baseline values use the original web-scraped SDCNL labels and remain preserved for transparent comparison.

</details>

## Speech Branch
Official EATD participant-level validation:
- Accuracy: 63.29%
- Balanced Accuracy: 63.44%
- Precision: 21.88%
- Recall: 63.64%
- F1: 32.56%
- ROC-AUC: 59.22%

EATD is a depression/distress corpus, not a direct suicide-risk-labelled speech dataset.

## Evaluation Note
The high text result is a **label-corrected SDCNL evaluation**, following the same general idea used by the SDCNL paper: unsupervised embedding clustering with confidence-based label correction. It must be distinguished from the standard original-web-label baseline.

## Final Verification
Current result: **56/56 checks PASS, 0 failed**.

The automated project suite checks:
- required files
- Python source syntax
- SDCNL row counts and labels
- zero normalized train-test text overlap
- standard baseline metrics
- enhanced text-model artifacts and metrics
- speech validation metrics
- six system-design diagrams
- text inference
- real EATD WAV speech inference
- late-fusion output

## Installation

Recommended: Python 3.11.

```bash
python -m pip install -r requirements.txt
```

On Windows you can also run:

```
SETUP_PROJECT.bat
```

## Run

```bash
python -m streamlit run app.py
```

or double-click:

```
START_PROJECT.bat
```

Then open:

```
http://localhost:8501
```

## Verify

```
VERIFY_PROJECT.bat
```

or:

```bash
python tests/run_project_tests.py
```

## Dataset Setup

Large raw audio datasets are intentionally **not stored in this GitHub repository** because of GitHub file-size/repository-size limits.

The project uses:
- **SDCNL** for suicide/depression text classification
- **EATD-Corpus** for depression/distress speech analysis

After obtaining the datasets, place them under the local `Datasets/` directory using the folder structure expected by the preprocessing/training scripts.

Important: SDCNL and EATD contain different participants and different targets. They must **not** be merged row-by-row.

## Fusion

```
combined_score = 0.70 * P_text + 0.30 * P_speech
```

Bands:
- Low: < 0.40
- Moderate: 0.40 to < 0.70
- High: >= 0.70

These are transparent academic prototype settings, not clinically calibrated thresholds.

## Important Files
- `app.py` — Streamlit interface
- `src/inference_pipeline.py` — unified fast inference
- `src/sdcnl_label_correction_experiment.py` — offline BERT → UMAP → GMM label-correction experiment
- `src/finalize_enhanced_fast_text_model.py` — final fast enhanced text model
- `models/enhanced_text_model/` — production text-model artifacts
- `models/sdcnl_label_correction_2026-10-31/` — offline label-correction artifacts
- `src/train_text_svm_baseline.py` — standard baseline
- `src/train_text_lstm.py` — BiLSTM baseline
- `src/train_text_bert_embeddings.py` — frozen BERT baseline
- `src/train_speech_sample_model.py` — speech model
- `config/fusion_config.json` — decision configuration
- `tests/run_project_tests.py` — automated verification
- `diagrams/Thanmayi_Final_System_Design_Diagrams.drawio` — system-design diagrams
- `Thanmayi_Final_Project_Report.pdf` — final project report

## Scientific Limitations
- SDCNL web labels are noisy; label-corrected metrics and original-web-label metrics are different evaluation settings.
- SDCNL and EATD are not participant-matched.
- EATD predicts SDS-derived depression/distress rather than direct suicide risk.
- Fusion weights are prototype settings, not learned clinical thresholds.
- The system is a research-screening prototype, not a clinical diagnostic tool.
