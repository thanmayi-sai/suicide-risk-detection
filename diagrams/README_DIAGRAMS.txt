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
- Text branch: TF-IDF + calibrated Linear SVM is the current best verified text model.
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
