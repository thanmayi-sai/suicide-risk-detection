THANMAYI FINAL SYSTEM DESIGN DIAGRAMS
=====================================
Project: Suicide-Risk Detection Using NLP-Based Text and Speech Analysis

This package contains six verified diagram pages inside one editable draw.io file:
1. Architecture - UML Component
2. System Flow Chart
3. Logical Diagram
4. Behavioural - Activity Diagram
5. Conceptual ER Diagram
6. DFD Level 1

Current implementation alignment:
- Primary text branch: word+character TF-IDF + calibrated LinearSVC trained on SDCNL UMAP-GMM confidence-corrected labels.
- Primary text label-corrected evaluation: Accuracy 85.75%, Precision 93.98%, Recall 78.00%, F1 85.25%, ROC-AUC 95.04%.
- Standard original-web-label baseline is retained separately: TF-IDF + calibrated Linear SVM accuracy 72.56%.
- Offline BERT label-corrected research experiment: 93.14% accuracy; research comparison only.
- Speech branch: single-audio acoustic Logistic Regression selected with participant-grouped CV.
- Speech outcome: EATD SDS-derived depression/distress signal, not direct suicide risk.
- Combined decision: transparent prototype late fusion of branch scores; not clinically/jointly calibrated.
- SDCNL and EATD are not merged row-by-row.

Automated structural verification:
- 6 required pages present
- 0 broken connections
- 0 box overlaps
- 0 out-of-bounds boxes
- old/obsolete architecture terms absent

See DIAGRAM_VERIFICATION_REPORT.txt / .json for the exact audit.
