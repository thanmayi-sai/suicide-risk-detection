# FINAL PROJECT VERIFICATION REPORT — 31 OCTOBER 2026

## Project
Suicide-Risk Detection Using NLP-Based Text and Speech Analysis

## Final project location
C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000

## Backup created before accuracy enhancement
C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000_BACKUP_BEFORE_ACCURACY_TUNING_2026-10-31

## Accuracy enhancement completed
The standard original-web-label baseline is preserved:
- TF-IDF + calibrated Linear SVM accuracy: 72.56%
- Precision: 71.92%
- Recall: 75.65%
- F1: 73.74%
- ROC-AUC: 78.22%

The project now implements an SDCNL-style unsupervised label-correction stage:
- Frozen BERT mean-pooled embeddings
- UMAP dimensionality reduction
- Gaussian Mixture Model
- confidence threshold tau = 0.90

Primary fast production/demo text model:
- Word TF-IDF (1–2 grams)
- Character TF-IDF (3–5 grams)
- Calibrated LinearSVC
- trained on SDCNL confidence-corrected training labels

Primary label-corrected evaluation:
- Accuracy: 85.75%
- Precision: 93.98%
- Recall: 78.00%
- F1: 85.25%
- ROC-AUC: 95.04%
- Confusion Matrix: [[169,10],[44,156]]

Separate offline label-corrected BERT research experiment:
- Accuracy: 93.14%
- Precision: 95.31%
- Recall: 91.50%
- F1: 93.37%
- ROC-AUC: 98.53%

Important: 85.75% and 93.14% are SDCNL-style label-corrected evaluations. They do not replace or hide the standard original-web-label baseline.

## Speech branch
Official EATD participant-level validation:
- Accuracy: 63.29%
- Balanced Accuracy: 63.44%
- Precision: 21.88%
- Recall: 63.64%
- F1: 32.56%
- ROC-AUC: 59.22%

## Final inference verification
Verified example text:
I am feeling very stressed and overwhelmed today.

Text-only:
- P_text: 0.9878
- Screening level: High

Speech-only using:
Datasets\validation\10\negative_out.wav
- P_speech: 0.2816
- Screening level: Low

Text + Speech:
- Combined evidence score: 0.7759
- Screening level: High

## Automated test result
- Passed: 56
- Failed: 0
- Overall: PASS

The suite verifies:
- required project files
- syntax
- original SDCNL baseline
- enhanced text artifacts
- enhanced accuracy >= 85%
- enhanced ROC-AUC >= 95%
- offline label-correction research result
- original and corrected evaluation separation
- dataset integrity
- zero normalized train-test overlap
- speech metrics
- six diagrams
- enhanced text inference
- real EATD WAV inference
- fusion score and screening band

## Diagrams
Six required diagrams verified.
- Broken connections: 0
- Box overlaps: 0
- Out-of-bounds boxes: 0
- Overall structural verification: PASS

Diagram source and ZIP were updated to the enhanced text model and completed speech model.

## Final documentation
Updated:
- README.md
- PROJECT_RUN_GUIDE_BEGINNER.docx
- PROJECT_RUN_GUIDE_BEGINNER.pdf
- Thanmayi_Final_Project_Report.docx
- Thanmayi_Final_Project_Report.pdf
- config/fusion_config.json
- app.py
- diagrams
- output screenshots
- automated tests

## App startup
Updated Streamlit application successfully started on localhost and was closed after verification.

## Final status
IMPLEMENTATION-READY RESEARCH PROTOTYPE: COMPLETE.

Scientific limitations remain explicit:
- label-corrected text metrics and original-web-label metrics are different evaluation settings
- EATD is depression/distress speech data, not direct suicide-risk speech ground truth
- SDCNL and EATD are not participant-matched
- late fusion is a transparent research prototype and not clinically calibrated
