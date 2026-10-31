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

1. **Text only** — SDCNL text input using TF-IDF + calibrated Linear SVM.
2. **Speech only** — EATD acoustic feature analysis using Logistic Regression.
3. **Text + Speech** — transparent late-fusion research decision layer.

> This project is for academic/research screening only. It is not a clinical diagnostic system.

## Verified Results

### Text branch
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BiLSTM | 59.37% | 57.77% | 75.13% | 65.32% | 63.58% |
| Frozen BERT + Logistic Regression | 64.12% | 63.77% | 68.39% | 66.00% | 72.27% |
| **TF-IDF + Calibrated Linear SVM** | **72.56%** | **71.92%** | **75.65%** | **73.74%** | **78.22%** |

### Speech branch
Official EATD participant-level validation:
- Accuracy: 63.29%
- Balanced Accuracy: 63.44%
- Precision: 21.88%
- Recall: 63.64%
- F1: 32.56%
- ROC-AUC: 59.22%

## Final Verification
Automated project verification: **37/37 checks passed**.

The checks cover:
- required files
- source syntax
- dataset row counts
- binary labels
- zero normalized SDCNL train-test overlap
- stored metrics
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

Expected result:

```
Passed: 37
Failed: 0
Overall: PASS
```

## Dataset Setup

Large raw audio datasets are intentionally **not stored in this GitHub repository** because of GitHub file-size/repository-size limits.

The project uses:
- **SDCNL** for suicide/depression text classification
- **EATD-Corpus** for depression/distress speech analysis

After obtaining the datasets, place them under the local `Datasets/` directory using the folder structure expected by the preprocessing/training scripts.

Important: SDCNL and EATD contain different participants and different targets. They must **not** be merged row-by-row.

## Fusion
Prototype combined mode uses:

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
- `src/inference_pipeline.py` — unified inference
- `src/train_text_svm_baseline.py` — final text baseline training
- `src/train_text_lstm.py` — BiLSTM experiment
- `src/train_text_bert_embeddings.py` — frozen BERT baseline
- `src/train_speech_sample_model.py` — speech model training
- `config/fusion_config.json` — prototype decision configuration
- `tests/run_project_tests.py` — automated verification
- `diagrams/Thanmayi_Final_System_Design_Diagrams.drawio` — system-design diagrams
- `Thanmayi_Final_Project_Report.pdf` — final project report

## Scientific Limitation
The text and speech datasets are not paired at participant level. EATD provides a depression/distress target rather than direct suicide-risk ground truth. Therefore the repository does not claim a clinically calibrated final multimodal accuracy.
