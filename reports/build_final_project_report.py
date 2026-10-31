
from pathlib import Path
import json, re, textwrap, math
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_BREAK
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
                                Image as RLImage, KeepTogether, Preformatted)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
ASSET=ROOT/"reports"/"final_report_assets"
DOCX_OUT=ROOT/"Thanmayi_Final_Project_Report.docx"
PDF_OUT=ROOT/"Thanmayi_Final_Project_Report.pdf"

TITLE="Suicide-Risk Detection Using NLP-Based Text and Speech Analysis"
GUIDE="Dr. G. Rajendra"
GUIDE_DESIG="Associate Professor"
STUDENTS=[
    ("S. Thanmayi","23761A1251"),
    ("Ch. Venu Madhav","24765A1201"),
    ("M. Samuel Babu","23761A1235"),
]
YEAR="2026-2027"
COLLEGE="LAKIREDDY BALI REDDY COLLEGE OF ENGINEERING"
DEPT="DEPARTMENT OF INFORMATION TECHNOLOGY"

# ----------------- DOCX HELPERS -----------------
def set_cell_margins(cell, top=80, start=100, bottom=80, end=100):
    tc=cell._tc
    tcPr=tc.get_or_add_tcPr()
    tcMar=tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar=OxmlElement("w:tcMar"); tcPr.append(tcMar)
    for m,v in [("top",top),("start",start),("bottom",bottom),("end",end)]:
        node=tcMar.find(qn("w:"+m))
        if node is None:
            node=OxmlElement("w:"+m); tcMar.append(node)
        node.set(qn("w:w"),str(v)); node.set(qn("w:type"),"dxa")

def add_page_border(section):
    sectPr=section._sectPr
    pgBorders=sectPr.find(qn("w:pgBorders"))
    if pgBorders is None:
        pgBorders=OxmlElement("w:pgBorders")
        pgBorders.set(qn("w:offsetFrom"),"page")
        sectPr.append(pgBorders)
    for edge in ("top","left","bottom","right"):
        e=OxmlElement("w:"+edge)
        e.set(qn("w:val"),"single")
        e.set(qn("w:sz"),"10")
        e.set(qn("w:space"),"20")
        e.set(qn("w:color"),"000000")
        pgBorders.append(e)

def add_page_number(section, fmt="decimal"):
    footer=section.footer
    p=footer.paragraphs[0]
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=p.add_run()
    fld=OxmlElement("w:fldSimple"); fld.set(qn("w:instr"),"PAGE")
    run._r.addnext(fld)
    sectPr=section._sectPr
    pgNum=sectPr.find(qn("w:pgNumType"))
    if pgNum is None:
        pgNum=OxmlElement("w:pgNumType"); sectPr.append(pgNum)
    pgNum.set(qn("w:fmt"),fmt)

def set_columns(section,n=2,space_twips=360):
    sectPr=section._sectPr
    cols=sectPr.find(qn("w:cols"))
    if cols is None:
        cols=OxmlElement("w:cols"); sectPr.append(cols)
    cols.set(qn("w:num"),str(n))
    cols.set(qn("w:space"),str(space_twips))

def set_run_font(run,name="Times New Roman",size=12,bold=None,italic=None):
    run.font.name=name; run.font.size=Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"),name)
    if bold is not None: run.bold=bold
    if italic is not None: run.italic=italic

def add_center(doc,text,size=12,bold=False,space_before=0,space_after=6):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(space_before); p.paragraph_format.space_after=Pt(space_after)
    r=p.add_run(text); set_run_font(r,size=size,bold=bold)
    return p

def add_body(doc,text,bold_prefix=None):
    p=doc.add_paragraph()
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing=1.15
    p.paragraph_format.space_after=Pt(6)
    p.paragraph_format.first_line_indent=Inches(.28)
    if bold_prefix and text.startswith(bold_prefix):
        r=p.add_run(bold_prefix); set_run_font(r,size=12,bold=True)
        r=p.add_run(text[len(bold_prefix):]); set_run_font(r,size=12)
    else:
        r=p.add_run(text); set_run_font(r,size=12)
    return p

def add_bullet(doc,text):
    p=doc.add_paragraph(style="List Bullet")
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after=Pt(4)
    for r in p.runs: set_run_font(r,size=11.5)
    if not p.runs:
        r=p.add_run(text); set_run_font(r,size=11.5)
    else:
        p.runs[0].text=text
    return p

def add_heading(doc,text,level=1):
    p=doc.add_paragraph()
    p.paragraph_format.space_before=Pt(6); p.paragraph_format.space_after=Pt(8)
    r=p.add_run(text); set_run_font(r,size=13 if level==1 else 12,bold=True)
    return p

def chapter_page(doc,n):
    doc.add_page_break()
    for _ in range(9): doc.add_paragraph()
    add_center(doc,f"CHAPTER - {n}",size=16,bold=True)
    doc.add_page_break()

def add_picture(doc,path,caption,width=6.2):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    run=p.add_run(); run.add_picture(str(path),width=Inches(width))
    c=doc.add_paragraph(); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=c.add_run(caption); set_run_font(r,size=10,bold=True)
    return c

def add_code(doc,code,title=None,max_lines=None):
    if title: add_heading(doc,title,2)
    if max_lines: code="\n".join(code.splitlines()[:max_lines])
    p=doc.add_paragraph()
    p.paragraph_format.left_indent=Inches(.15); p.paragraph_format.right_indent=Inches(.15)
    p.paragraph_format.space_after=Pt(5)
    r=p.add_run(code); set_run_font(r,name="Consolas",size=7.5)
    shading=OxmlElement("w:shd"); shading.set(qn("w:fill"),"F4F4F4")
    p._p.get_or_add_pPr().append(shading)

# ----------------- CREATE DOCX -----------------
doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
sec.top_margin=Inches(.68); sec.bottom_margin=Inches(.68); sec.left_margin=Inches(.75); sec.right_margin=Inches(.75)
add_page_border(sec)

styles=doc.styles
styles["Normal"].font.name="Times New Roman"; styles["Normal"].font.size=Pt(12)
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"),"Times New Roman")
if "CodeSmall" not in styles:
    st=styles.add_style("CodeSmall",WD_STYLE_TYPE.PARAGRAPH); st.font.name="Consolas"; st.font.size=Pt(7.5)

# Cover
for _ in range(2): doc.add_paragraph()
add_center(doc,TITLE,size=17,bold=True,space_after=8)
add_center(doc,"A Project Report",size=14,bold=False,space_after=18)
add_center(doc,"Submitted in partial fulfilment of requirements for the award of the degree of",size=11)
add_center(doc,"BACHELOR OF TECHNOLOGY",size=15,bold=True)
add_center(doc,"IN",size=12,bold=True)
add_center(doc,"INFORMATION TECHNOLOGY",size=14,bold=True,space_after=15)
add_center(doc,"Submitted by",size=11,bold=True)
for n,rn in STUDENTS: add_center(doc,f"{n} ({rn})",size=12,bold=True,space_after=2)
doc.add_paragraph()
add_center(doc,"Under the Esteemed Guidance of",size=11,bold=True)
add_center(doc,GUIDE,size=13,bold=True,space_after=1)
add_center(doc,GUIDE_DESIG,size=11)
add_center(doc,"Dept. of IT, LBRCE",size=11)
for _ in range(2): doc.add_paragraph()
add_center(doc,DEPT,size=13,bold=True)
add_center(doc,COLLEGE,size=14,bold=True)
add_center(doc,"(AUTONOMOUS)",size=11,bold=True)
add_center(doc,"An ISO 21001:2018, 14001:2015, 50001:2018 Certified Institution",size=9.5)
add_center(doc,"Approved by AICTE, New Delhi and Affiliated to JNTUK, Kakinada",size=9.5)
add_center(doc,"L.B. REDDY NAGAR, MYLAVARAM, NTR DIST., A.P.-521 230",size=9.5)
add_center(doc,YEAR,size=11,bold=True)

# Certificate
doc.add_page_break()
add_center(doc,COLLEGE,size=14,bold=True)
add_center(doc,"(AUTONOMOUS)",size=11,bold=True)
add_center(doc,"Accredited by NAAC & NBA (Under Tier - I)",size=9.5)
add_center(doc,"An ISO 21001:2018, 14001:2015, 50001:2018 Certified Institution",size=9.5)
add_center(doc,"Approved by AICTE, New Delhi and Affiliated to JNTUK, Kakinada",size=9.5)
add_center(doc,"L.B. REDDY NAGAR, MYLAVARAM, NTR DIST., A.P.-521 230",size=9.5)
doc.add_paragraph()
add_center(doc,DEPT,size=13,bold=True)
doc.add_paragraph()
add_center(doc,"CERTIFICATE",size=15,bold=True)
cert=(f"This is to certify that {STUDENTS[0][0]} ({STUDENTS[0][1]}), {STUDENTS[1][0]} ({STUDENTS[1][1]}) "
      f"and {STUDENTS[2][0]} ({STUDENTS[2][1]}), students of IV B. Tech (Information Technology), "
      f"have successfully completed the project titled “{TITLE}” at Lakireddy Bali Reddy College of Engineering "
      f"during the Academic Year {YEAR}. This project report is submitted as partial fulfilment for the award of "
      f"the degree of Bachelor of Technology (Information Technology).")
add_body(doc,cert)
for _ in range(4): doc.add_paragraph()
t=doc.add_table(rows=2,cols=2); t.alignment=WD_TABLE_ALIGNMENT.CENTER
vals=[("Project Guide","Head of the Department"),(f"{GUIDE}\n{GUIDE_DESIG}", "Dr. D. Ratna Kishore\nProfessor & HOD, IT")]
for i,row in enumerate(vals):
    for j,v in enumerate(row):
        t.cell(i,j).text=v; t.cell(i,j).vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in t.cell(i,j).paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: set_run_font(r,size=10.5,bold=(i==0))
for _ in range(2): doc.add_paragraph()
t2=doc.add_table(rows=2,cols=2); t2.alignment=WD_TABLE_ALIGNMENT.CENTER
vals=[("Project Co Ordinator name","External Examiner"),("Mr. M. Vijay Kumar\nSr Assistant Professor","")]
for i,row in enumerate(vals):
    for j,v in enumerate(row):
        t2.cell(i,j).text=v
        for p in t2.cell(i,j).paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: set_run_font(r,size=10.5,bold=(i==0))

# Acknowledgement
doc.add_page_break(); add_center(doc,"ACKNOWLEDGEMENT",size=14,bold=True)
acks=[
"Behind every achievement lies the support and encouragement of many people. We express our sincere gratitude to all those who contributed directly or indirectly to the successful completion of this project.",
"We express our special gratitude to Dr. K. Appa Rao, Principal, Lakireddy Bali Reddy College of Engineering, for providing the academic environment and facilities required to carry out this work. We also thank all teaching and non-teaching staff members of the Department of Information Technology for their cooperation.",
"We are highly obliged to Dr. D. Ratna Kishore, Head of the Department of Information Technology, for his encouragement and support throughout the project.",
f"It is our privilege to express heartfelt gratitude to our guide, {GUIDE}, {GUIDE_DESIG}, Department of Information Technology, for valuable guidance, technical suggestions, careful review and continuous support during all phases of the project.",
"We are also thankful to the project coordinator, Mr. M. Vijay Kumar, Sr Assistant Professor, Department of IT, for coordinating the project reviews and providing systematic guidance.",
"Finally, we express our gratitude to our parents and friends for their constant encouragement and support."
]
for x in acks: add_body(doc,x)

# Declaration
doc.add_page_break(); add_center(doc,"CANDIDATE’S DECLARATION",size=14,bold=True)
add_body(doc,f"We hereby declare that the project report entitled “{TITLE}” is a record of our work carried out during the academic year {YEAR} under the guidance of {GUIDE}, {GUIDE_DESIG}, Department of Information Technology, and submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology (B. Tech) in Information Technology.")
add_body(doc,"The matter embodied in this project is genuine work carried out by us and has not been submitted earlier to this university or any other university for the award of any degree, diploma or prize.")
for _ in range(6): doc.add_paragraph()
add_center(doc,"PROJECT ASSOCIATES",size=11,bold=True)
for n,rn in STUDENTS: add_center(doc,f"{n} ({rn})",size=11,bold=True,space_after=1)

# Index
doc.add_page_break(); add_center(doc,"INDEX",size=14,bold=True)
index_rows=[
("LIST OF FIGURES","I"),("ACRONYMS AND ABBREVIATIONS","II"),("ABSTRACT","III"),
("1. INTRODUCTION","1-3"),("    1.1 Introduction","1-2"),("    1.2 Existing System","2"),("    1.3 Proposed System","3"),
("2. LITERATURE SURVEY","4-5"),("3. SYSTEM SPECIFICATIONS","6-9"),
("    3.1 Software Requirements","6-7"),("    3.2 Hardware Requirements","7"),("    3.3 Functional Requirements","8"),("    3.4 Non-Functional Requirements","9"),
("4. METHODOLOGY","10-13"),("5. DESIGN VIEW","14-18"),("6. IMPLEMENTATION","19-29"),
("7. TESTING","30-31"),("8. RESULTS AND EVALUATION","32-35"),("9. CONCLUSION","36"),
("10. FUTURE ENHANCEMENT","37"),("REFERENCES","38-39"),("APPENDIX","")
]
tbl=doc.add_table(rows=1,cols=2); tbl.alignment=WD_TABLE_ALIGNMENT.CENTER; tbl.style="Table Grid"
tbl.cell(0,0).text="Contents"; tbl.cell(0,1).text="Page No"
for a,b in index_rows:
    c=tbl.add_row().cells; c[0].text=a; c[1].text=b
for row in tbl.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            for r in p.runs: set_run_font(r,size=10,bold=(row is tbl.rows[0]))
        set_cell_margins(cell)

# List figures
doc.add_page_break(); add_center(doc,"LIST OF FIGURES",size=14,bold=True)
figs=[
("4.1","System Architecture Diagram","10"),
("5.1","System Flow Chart","14"),("5.2","Logical Diagram","15"),("5.3","Activity Diagram","16"),
("5.4","Conceptual ER Diagram","17"),("5.5","DFD Level 1","18"),
("8.1","Text Model Performance Comparison","32"),("8.2","Standard Text SVM Confusion Matrix","33"),
("8.3","Enhanced Label-Corrected Text Confusion Matrix","33"),("8.4","Speech Model Confusion Matrix","34"),
("8.5","Text-only Output","35"),("8.6","Speech-only Output","35"),
("8.7","Text + Speech Output","36"),("8.8","Automated Test Summary","36")
]
tb=doc.add_table(rows=1,cols=3); tb.style="Table Grid"; tb.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,v in enumerate(["Figure No","Description","Page No"]): tb.cell(0,j).text=v
for row in figs:
    c=tb.add_row().cells
    for j,v in enumerate(row): c[j].text=v
for row in tb.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: set_run_font(r,size=9.5,bold=(row is tb.rows[0]))

# Acronyms
doc.add_page_break(); add_center(doc,"ACRONYMS AND ABBREVIATIONS",size=14,bold=True)
acrs=[
("AI","Artificial Intelligence"),("NLP","Natural Language Processing"),
("SDCNL","Suicide and Depression Classification with Noisy Labels"),
("EATD","Emotional Audio-Textual Depression Corpus"),("TF-IDF","Term Frequency-Inverse Document Frequency"),
("SVM","Support Vector Machine"),("BERT","Bidirectional Encoder Representations from Transformers"),
("BiLSTM","Bidirectional Long Short-Term Memory"),("LR","Logistic Regression"),
("MFCC","Mel-Frequency Cepstral Coefficients"),("RMS","Root Mean Square Energy"),
("ZCR","Zero-Crossing Rate"),("ROC-AUC","Receiver Operating Characteristic - Area Under Curve"),
("CV","Cross-Validation"),("UI","User Interface")
]
ta=doc.add_table(rows=0,cols=2); ta.style="Table Grid"; ta.alignment=WD_TABLE_ALIGNMENT.CENTER
for a,b in acrs:
    c=ta.add_row().cells; c[0].text=a; c[1].text=b
    for cell in c:
        for p in cell.paragraphs:
            for r in p.runs:set_run_font(r,size=10)

# Abstract
doc.add_page_break(); add_center(doc,"ABSTRACT",size=14,bold=True)
abstract=("Suicide-risk identification is a challenging research problem because warning signals can appear in written language, speech characteristics, or both. "
"This project presents a multimodal research screening framework that combines Natural Language Processing based text analysis with acoustic speech analysis. "
"The text branch uses the SDCNL dataset and evaluates a BiLSTM, frozen BERT mean-pooled embeddings with Logistic Regression, and TF-IDF bigram features with a calibrated Linear SVM. "
"After leakage-free preprocessing, the final SDCNL training set contains 1,513 records and the held-out test set contains 379 records, with zero normalized train-test text overlap. "
"The standard original-web-label baseline, TF-IDF with calibrated Linear SVM, achieved 72.56% accuracy. To address noisy web labels, an enhanced SDCNL-style label-correction stage was implemented using frozen BERT embeddings, UMAP, Gaussian Mixture Model clustering and a confidence threshold of 0.90. The final fast word+character TF-IDF + calibrated LinearSVC trained on the corrected labels achieved 85.75% accuracy, 93.98% precision, 78.00% recall, 85.25% F1-score and 95.04% ROC-AUC on the label-corrected evaluation. A separate offline BERT label-corrected research experiment reached 93.14% accuracy. "
"The speech branch uses the EATD-Corpus as a complementary depression/distress source. Speech is processed at mono 16 kHz with peak normalization, and acoustic features including MFCC, pitch, RMS energy, zero-crossing rate, spectral centroid and duration are extracted. "
"A participant-grouped Logistic Regression speech model achieved 63.29% participant-level validation accuracy, 63.44% balanced accuracy and 59.22% ROC-AUC. "
"Since SDCNL and EATD contain different participants and different targets, they are not merged row-by-row. The final application keeps P_text and P_speech separate and applies a transparent prototype late-fusion decision layer for Low, Moderate and High research-screening levels. "
"The Streamlit implementation supports text-only, speech-only and combined input modes, and the final automated verification suite passed all 56 implemented checks. "
"The system is intended for academic research screening and not for clinical diagnosis.")
add_body(doc,abstract)

# New body section with Arabic page numbers
sec2=doc.add_section(WD_SECTION.NEW_PAGE); sec2.page_width=Inches(8.27); sec2.page_height=Inches(11.69)
sec2.top_margin=Inches(.68); sec2.bottom_margin=Inches(.68); sec2.left_margin=Inches(.75); sec2.right_margin=Inches(.75)
add_page_border(sec2); add_page_number(sec2,"decimal")

# Chapter 1
chapter_page(doc,1)
add_center(doc,"1.1. INTRODUCTION:",size=13,bold=True)
for p in [
"Suicide and self-harm related risk is a major mental-health concern, and early identification of warning signals can support timely human intervention. Conventional assessment depends on professional interviews and validated clinical instruments. These remain essential for diagnosis, but they are difficult to scale continuously across large online populations and may not capture every individual who is reluctant to disclose distress directly.",
"Digital platforms have created new sources of behavioural and linguistic information. Social-media posts, forum messages and other online text can contain patterns associated with hopelessness, distress, self-harm references and changes in emotional expression. Natural Language Processing makes it possible to represent and classify such text automatically, while speech processing can capture complementary acoustic characteristics such as pitch variation, energy, timing and spectral patterns.",
"This project therefore studies a multimodal research-screening architecture with independent text and speech branches. The text branch is trained on SDCNL suicide/depression text, while the speech branch uses EATD depression/distress speech. Because the two datasets do not contain the same people and do not share an identical target label, the project does not perform invalid row-wise multimodal training. Instead, each branch is evaluated separately and the resulting evidence scores are combined only at the final prototype decision layer."
]: add_body(doc,p)
doc.add_page_break()
add_center(doc,"1.2. EXISTING SYSTEM:",size=13,bold=True)
for x in [
"Traditional screening relies on interviews and questionnaires conducted by trained mental-health professionals. These methods are clinically meaningful but require time and expert availability.",
"Many automated text systems use only one representation or one classifier. Their results can be affected by noisy labels, duplicate records, train-test leakage and poor probability calibration.",
"Speech-based mental-health systems often require specialized corpora. Public speech datasets with direct suicide-risk labels are scarce, which makes it unsafe to assume that a depression corpus is equivalent to a suicide-risk corpus.",
"Some multimodal studies combine paired text and audio from the same participants. Such approaches cannot be reproduced by simply concatenating unrelated SDCNL and EATD rows because the participants and labels do not correspond."
]: add_bullet(doc,x)
doc.add_page_break()
add_center(doc,"1.3. PROPOSED SYSTEM:",size=13,bold=True)
for x in [
"Leakage-aware text pipeline using the cleaned SDCNL training and fixed held-out test split.",
"Comparison of BiLSTM, frozen BERT embeddings with Logistic Regression, and standard TF-IDF + calibrated Linear SVM.",
"SDCNL-style unsupervised label correction using BERT embeddings → UMAP → Gaussian Mixture Model with confidence threshold τ=0.90.",
"Primary fast enhanced classifier using word+character TF-IDF + calibrated LinearSVC trained on the corrected SDCNL labels.",
"Independent speech pipeline using EATD acoustic features with participant-grouped cross-validation.",
"Single-audio speech inference with MFCC, pitch, RMS energy, ZCR, spectral centroid and duration features.",
"Late decision-level fusion of P_text and P_speech only, rather than invalid feature-level merging of unmatched datasets.",
"Streamlit interface supporting Text Only, Speech Only and Text + Speech modes.",
"Low / Moderate / High research-screening display with clear non-diagnostic disclaimer.",
"Automated verification covering files, syntax, dataset integrity, model metrics, diagrams and end-to-end inference."
]: add_bullet(doc,x)

# Chapter 2
chapter_page(doc,2)
add_center(doc,"2. LITERATURE SURVEY",size=13,bold=True)
lit=[
("Haque, Reddi and Giallanza (ICANN 2021)","proposed SDCNL for suicide-versus-depression text classification and addressed noisy web labels using unsupervised label correction. Their work is directly relevant to the text dataset used in this project and motivates careful treatment of noisy labels and evaluation leakage."),
("Shen, Yang and Lin (ICASSP 2022)","introduced the EATD-Corpus and a GRU/BiLSTM-based depression detection framework using emotional audio and textual responses. EATD contains audio and transcripts from 162 volunteers and provides the speech/distress source used in this project."),
("Pokrywka, Kaczmarek and Gorzelańczyk (IEEE BigData 2024)","evaluated modern transformer configurations for suicide-risk classification on social-media text and demonstrated the strong potential of contextual language models in risk classification."),
("Jin, Liu and Dai (2025)","studied multimodal speech and text models for adolescent suicide-risk detection using paired recordings and transcripts. Their findings support multimodal analysis, but their paired-data setting differs from this project because SDCNL and EATD are unmatched."),
("Devlin et al. (2019)","introduced BERT, a bidirectional transformer architecture that enables contextual representation of language. In this project BERT is used as a frozen mean-pooled embedding baseline rather than as a fully fine-tuned final model."),
("Classical calibrated machine learning","remains competitive on limited text datasets. The project therefore compares deep-learning representations against a calibrated linear SVM with TF-IDF features instead of assuming a transformer will automatically provide the best held-out performance.")
]
for h,b in lit:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(7)
    r=p.add_run(h+": "); set_run_font(r,size=11.5,bold=True)
    r=p.add_run(b); set_run_font(r,size=11.5)
doc.add_page_break()
add_heading(doc,"Research Gap",1)
for p in [
"Publicly available suicide-related text datasets and public depression-related speech datasets usually do not describe the same participants. A system that directly merges such datasets can create an artificial multimodal relationship that does not exist in the original data.",
"Another recurring concern is data leakage. Duplicate or near-duplicate text across training and testing can inflate reported performance. This project explicitly checks normalized overlap and retains a leakage-free held-out test set.",
"The project therefore focuses on scientifically supportable modular evaluation: direct suicide-related text classification on SDCNL, complementary depression/distress speech modelling on EATD, and transparent late fusion only at the decision layer."
]: add_body(doc,p)

# Chapter 3
chapter_page(doc,3)
add_center(doc,"3. SYSTEM SPECIFICATIONS",size=13,bold=True)
add_heading(doc,"3.1 SOFTWARE REQUIREMENTS:",1)
soft=[
("Python 3.11","Primary programming environment used for preprocessing, model training, evaluation, inference and testing."),
("NumPy","Numerical operations and feature arrays."),
("Pandas","CSV loading, cleaning, participant-level feature tables and evaluation data."),
("Scikit-learn","TF-IDF, LinearSVC, probability calibration, Logistic Regression, feature selection, cross-validation and metrics."),
("PyTorch","BiLSTM text baseline implementation."),
("Hugging Face Transformers","Frozen BERT embedding baseline."),
("Librosa","Audio loading and acoustic feature extraction."),
("SoundFile","Audio support."),
("Joblib","Saving and loading vectorizers and trained models."),
("Streamlit","Local browser-based user interface."),
("Matplotlib","Charts and evaluation figures."),
("UMAP-learn","Dimensionality reduction used in the SDCNL-style unsupervised label-correction experiment.")
]
for a,b in soft:
    p=doc.add_paragraph(); r=p.add_run(a+": ");set_run_font(r,size=11.5,bold=True); r=p.add_run(b);set_run_font(r,size=11.5)
doc.add_page_break()
add_heading(doc,"3.2 HARDWARE REQUIREMENTS:",1)
for x in ["64-bit Windows 10/11 system.","Intel Core i5/i7 or equivalent processor recommended.","Minimum 8 GB RAM; 16 GB preferred for development and model experimentation.","At least 10-20 GB free storage for datasets, model artifacts, reports and temporary files.","GPU is optional for the final app because the verified production inference pipeline can run on CPU; GPU is useful only for heavier deep-learning experiments."]: add_bullet(doc,x)
doc.add_page_break()
add_heading(doc,"3.3 FUNCTIONAL REQUIREMENTS:",1)
for x in ["Accept text-only, speech-only or combined input.","Validate that required inputs are present for the selected mode.","Transform text using the saved TF-IDF vectorizer and return calibrated P_text.","Load audio, normalize it and extract the same acoustic features used during speech training.","Return P_speech using the saved speech model and selected threshold.","Combine available branch scores according to the documented prototype fusion configuration.","Display screening level, branch scores, combined score, technical details and research disclaimer.","Provide reusable saved models so retraining is not required for every demonstration."]: add_bullet(doc,x)
doc.add_page_break()
add_heading(doc,"3.4 NON-FUNCTIONAL REQUIREMENTS:",1)
for x in ["Usability: a non-technical user should be able to run the application using helper batch files.","Reliability: model artifacts, file paths and dependencies are verified by an automated test suite.","Maintainability: training, inference, configuration and UI code are kept in separate files.","Transparency: text and speech branch scores are shown separately instead of hiding the fusion process.","Reproducibility: package versions are recorded in requirements.txt and final metrics are stored in JSON reports.","Safety of interpretation: the application clearly states that it is a research screening prototype, not a clinical diagnostic system."]: add_bullet(doc,x)

# Chapter 4
chapter_page(doc,4)
add_center(doc,"4. METHODOLOGY",size=13,bold=True)
add_heading(doc,"4.1. Overview of the Project:",1)
add_body(doc,"The methodology consists of two independently trained branches and a transparent decision layer. The text branch estimates suicide-related evidence from SDCNL text. The speech branch estimates depression/distress evidence from EATD speech. The two datasets are never treated as if they describe the same people.")
add_picture(doc,ASSET/"figure_4_1_architecture.png","Figure 4.1: System Architecture Diagram",5.9)
doc.add_page_break()
add_heading(doc,"4.2 Text Data Pre-processing",1)
for x in ["Remove unwanted index columns and create a model_text field from the cleaned source text.","Remove records that become empty after model-input cleaning.","Remove exact duplicate model_text + label records within each split.","Preserve the original binary labels.","Normalize case and whitespace for explicit train-test overlap checking.","Keep the final held-out test split isolated; do not train on combined-set-clean.csv because it contains test records."]: add_bullet(doc,x)
add_body(doc,"The final cleaned training set contains 1,513 rows and the fixed held-out test set contains 379 rows. The final normalized text overlap between training and testing is zero.")
add_heading(doc,"4.3 Text Model Development",1)
for x in ["BiLSTM baseline with train/validation split inside the training data.","TF-IDF unigrams/bigrams with calibrated Linear SVM.","Frozen bert-base-uncased mean-pooled embeddings followed by Logistic Regression.","All final comparisons are reported on the same 379-row leakage-free held-out test set."]: add_bullet(doc,x)
doc.add_page_break()
add_heading(doc,"4.3.1 SDCNL Label Correction and Enhanced Fast Text Model",1)
add_body(doc,"The SDCNL source labels are web-derived and can be noisy. To reproduce the label-correction idea described in the SDCNL study, the project uses frozen BERT mean-pooled embeddings, UMAP dimensionality reduction and a two-component Gaussian Mixture Model. When the maximum GMM membership probability is at least τ=0.90, the mapped cluster label is used as the corrected label. This produces a separate label-corrected evaluation and does not erase the standard original-web-label baseline.")
for x in [
"Frozen BERT mean-pooled embeddings provide semantic representations.",
"UMAP reduces the embedding space while preserving local structure.",
"Gaussian Mixture Model clustering provides soft cluster membership probabilities.",
"A confidence threshold τ=0.90 determines which labels are corrected.",
"The production/demo classifier is a fast word (1–2 gram) + character (3–5 gram) TF-IDF representation with calibrated LinearSVC trained on corrected training labels.",
"Final fast label-corrected metrics: Accuracy 85.75%, Precision 93.98%, Recall 78.00%, F1 85.25%, ROC-AUC 95.04%.",
"A separate offline BERT label-corrected Logistic Regression experiment achieved 93.14% accuracy and is retained as a research comparison only."
]: add_bullet(doc,x)
add_heading(doc,"4.4 Speech Pre-processing and Feature Extraction",1)
for x in ["Prefer *_out.wav files when available and usable.","Fallback to raw WAV only when the processed audio is unusable.","Load audio as mono at 16 kHz.","Apply peak normalization.","Extract 13 MFCC means and 13 MFCC standard deviations.","Extract pitch mean/std, RMS mean/std, ZCR mean/std, spectral centroid mean/std and duration."]: add_bullet(doc,x)
add_body(doc,"The EATD audit found 80 usable training participants and all 79 validation participants. The final audio-level feature tables contain 240 training rows and 237 validation rows.")
add_heading(doc,"4.5 Speech Model Selection",1)
add_body(doc,"The operational speech model is a single-audio Logistic Regression pipeline with feature selection. Candidate configurations were selected using participant-grouped cross-validation so that recordings from one participant never appear in both the training and validation folds. The official EATD validation split was used only after model and threshold selection.")
doc.add_page_break()
add_heading(doc,"4.6 Prototype Late Fusion and Decision Making",1)
for x in ["Text-only mode uses P_text directly.","Speech-only mode uses P_speech directly.","Combined mode computes 0.70 × P_text + 0.30 × P_speech.","Low: score < 0.40.","Moderate: 0.40 ≤ score < 0.70.","High: score ≥ 0.70."]: add_bullet(doc,x)
add_body(doc,"The fusion values are transparent research-prototype settings, not clinically calibrated thresholds. A valid learned multimodal fusion model would require paired text and speech with the same participants and the same target label.")
add_heading(doc,"4.7 Evaluation Metrics",1)
add_body(doc,"Accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix are used for model evaluation. Balanced accuracy is additionally emphasized for the imbalanced EATD participant validation set.")

# Chapter 5
chapter_page(doc,5)
add_center(doc,"5. DESIGN VIEW",size=13,bold=True)
designs=[
("5.1 System Flow Chart",ASSET/"figure_5_1_flow.png","Figure 5.1: System Flow Chart","The flow chart describes how the user selects an input mode, provides the required data, triggers branch inference, and receives the final research-screening result."),
("5.2 Logical Diagram",ASSET/"figure_5_2_logical.png","Figure 5.2: Logical Diagram","The logical view separates the input, processing, model and decision layers so that text and speech processing remain independent until the final decision stage."),
("5.3 Behavioural / Activity Diagram",ASSET/"figure_5_3_activity.png","Figure 5.3: Activity Diagram","The activity diagram represents the application workflow from input selection through validation, model execution, score combination and output display."),
("5.4 Conceptual ER Diagram",ASSET/"figure_5_4_er.png","Figure 5.4: Conceptual ER Diagram","The ER view is conceptual because the current prototype uses files and saved model artifacts rather than a relational database. It shows the logical relationship among input, branch results and screening result."),
("5.5 Data Flow Diagram",ASSET/"figure_5_5_dfd.png","Figure 5.5: DFD Level 1","The DFD shows user input entering the text and speech analysis processes, branch scores flowing into the decision process and the final result returning to the user.")
]
for i,(h,img,cap,desc) in enumerate(designs):
    add_heading(doc,h,1); add_body(doc,desc); add_picture(doc,img,cap,5.7)
    if i<len(designs)-1: doc.add_page_break()

# Chapter 6 Implementation
chapter_page(doc,6); add_center(doc,"6. IMPLEMENTATION",size=13,bold=True)
impl_intro=("The implementation is organized into dataset preprocessing, independent model training, saved model artifacts, a unified inference pipeline, a Streamlit user interface and automated verification. "
            "The following pages show the principal source-code components from the verified project. The complete executable source is stored in the project folder.")
add_body(doc,impl_intro)
files=[
("6.1 Text SVM Training","src/train_text_svm_baseline.py",95),
("6.2 BiLSTM Text Baseline","src/train_text_lstm.py",90),
("6.3 Frozen BERT Embedding Baseline","src/train_text_bert_embeddings.py",90),
("6.4 SDCNL Label-Correction Experiment","src/sdcnl_label_correction_experiment.py",120),
("6.5 Enhanced Fast Text Model","src/finalize_enhanced_fast_text_model.py",120),
("6.6 Speech Single-Audio Model","src/train_speech_sample_model.py",115),
("6.7 Unified Inference Pipeline","src/inference_pipeline.py",150),
("6.8 Streamlit User Interface","app.py",150),
("6.9 Automated Verification","tests/run_project_tests.py",170)
]
for i,(h,rel,nlines) in enumerate(files):
    pth=ROOT/rel
    code=pth.read_text(encoding="utf-8",errors="replace") if pth.exists() else "# File not found"
    add_heading(doc,h,1)
    add_code(doc,code,max_lines=nlines)
    if i<len(files)-1: doc.add_page_break()

# Chapter 7
chapter_page(doc,7); add_center(doc,"7. TESTING",size=13,bold=True)
add_body(doc,"Testing verifies the correct functioning and integrity of the complete research prototype. The project contains an automated verification suite that checks the required files, Python syntax, dataset counts, labels, text leakage, stored model metrics, diagrams and end-to-end branch inference.")
add_heading(doc,"7.1 Types of Testing Implemented",1)
for h,b in [
("Unit Testing","Individual preprocessing, text inference, speech feature extraction, model loading and score functions were checked."),
("Integration Testing","Interaction between saved models, the inference pipeline, fusion configuration and Streamlit application was verified."),
("Dataset Integrity Testing","SDCNL row counts, binary labels and normalized train-test overlap were checked."),
("Model Artifact Testing","Saved vectorizer, text model, speech model and metric files were verified."),
("System Testing","The complete application was started locally and text, speech and combined outputs were generated."),
("Diagram Verification","All six required diagrams were checked for required pages, broken connections, overlaps and obsolete architecture labels.")
]: 
    p=doc.add_paragraph(); r=p.add_run(h+": ");set_run_font(r,size=11.5,bold=True); r=p.add_run(b);set_run_font(r,size=11.5)
doc.add_page_break()
add_heading(doc,"7.2 Test Cases and Results",1)
tests=[
("TC1","Required project files","Files/models/configs available","Pass"),
("TC2","Python source syntax","All source files compile","Pass"),
("TC3","Text train rows","1,513 rows","Pass"),
("TC4","Text test rows","379 rows","Pass"),
("TC5","Train-test overlap","0 normalized overlaps","Pass"),
("TC6","Text inference","Probability in [0,1]","Pass"),
("TC7","Speech inference","Real EATD WAV returns probability","Pass"),
("TC8","Fusion output","Score in [0,1] + valid band","Pass"),
("TC9","Diagram verification","6 pages, no structural errors","Pass"),
("TC10","Complete automated suite","All implemented checks pass","Pass")
]
tt=doc.add_table(rows=1,cols=4); tt.style="Table Grid"; tt.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,v in enumerate(["ID","Test Description","Expected Output","Result"]): tt.cell(0,j).text=v
for row in tests:
    c=tt.add_row().cells
    for j,v in enumerate(row): c[j].text=v
for row in tt.rows:
    for cell in row.cells:
        for p in cell.paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:set_run_font(r,size=9,bold=(row is tt.rows[0]))

# Chapter 8
chapter_page(doc,8); add_center(doc,"8. RESULTS AND EVALUATION",size=13,bold=True)
add_body(doc,"The standard text baselines were evaluated against the original leakage-free SDCNL web labels. Because the SDCNL study identifies substantial web-label noise, this project also implements a separate SDCNL-style UMAP-GMM label-corrected evaluation. The primary fast enhanced model reaches 85.75% accuracy on the corrected-label evaluation, while the original-web-label baseline is preserved separately for transparency. The speech model is evaluated on the official EATD validation split with participant-level aggregation.")
add_picture(doc,ASSET/"figure_8_1_text_model_comparison.png","Figure 8.1: Text Model Performance Comparison",5.8)
doc.add_page_break()
add_heading(doc,"8.1 Text Model Results",1)
tr=doc.add_table(rows=1,cols=6); tr.style="Table Grid"; tr.alignment=WD_TABLE_ALIGNMENT.CENTER
for j,v in enumerate(["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]): tr.cell(0,j).text=v
for row in [
("BiLSTM","59.37%","57.77%","75.13%","65.32%","63.58%"),
("Frozen BERT + LR","64.12%","63.77%","68.39%","66.00%","72.27%"),
("TF-IDF + Calibrated SVM (original labels)","72.56%","71.92%","75.65%","73.74%","78.22%"),
("Enhanced Fast Model (label-corrected)","85.75%","93.98%","78.00%","85.25%","95.04%")]:
    c=tr.add_row().cells
    for j,v in enumerate(row): c[j].text=v
add_picture(doc,ASSET/"figure_8_2_text_cm.png","Figure 8.2: Standard Text SVM Confusion Matrix",4.3)
add_picture(doc,ASSET/"figure_8_2b_enhanced_text_cm.png","Figure 8.3: Enhanced Label-Corrected Text Model Confusion Matrix",4.3)
add_body(doc,"The enhanced metrics above use SDCNL-style corrected labels and must not be interpreted as the same evaluation target as the standard original-web-label baseline. A separate offline BERT label-corrected experiment achieved 93.14% accuracy, 95.31% precision, 91.50% recall, 93.37% F1 and 98.53% ROC-AUC.")
doc.add_page_break()
add_heading(doc,"8.2 Speech Model Results",1)
add_body(doc,"The participant-aggregated EATD validation result at the train-selected threshold of 0.38 is reported below. Accuracy alone is not sufficient because the validation set is imbalanced; therefore balanced accuracy, recall, F1 and ROC-AUC are also reported.")
sr=doc.add_table(rows=0,cols=2); sr.style="Table Grid"; sr.alignment=WD_TABLE_ALIGNMENT.CENTER
for a,b in [("Accuracy","63.29%"),("Balanced Accuracy","63.44%"),("Precision","21.88%"),("Recall","63.64%"),("F1-score","32.56%"),("ROC-AUC","59.22%")]:
    c=sr.add_row().cells;c[0].text=a;c[1].text=b
add_picture(doc,ASSET/"figure_8_3_speech_cm.png","Figure 8.4: Speech Model Confusion Matrix",4.5)
doc.add_page_break()
add_heading(doc,"8.3 Application Output",1)
for img,cap in [
(ROOT/"reports"/"OUTPUT_01_TEXT_ONLY.png","Figure 8.5: Text-only Output"),
(ROOT/"reports"/"OUTPUT_02_SPEECH_ONLY.png","Figure 8.6: Speech-only Output"),
(ROOT/"reports"/"OUTPUT_03_TEXT_AND_SPEECH.png","Figure 8.7: Text + Speech Output")]:
    if img.exists(): add_picture(doc,img,cap,5.9)
add_picture(doc,ASSET/"figure_8_7_test_summary.png","Figure 8.8: Final Automated Test Summary",5.2)
add_body(doc,"The final automated suite passed all 56 implemented checks. The generated UI outputs confirm that text-only, speech-only and combined modes execute using the saved enhanced text and speech model artifacts.")

# Conclusion
chapter_page(doc,9); add_center(doc,"9. CONCLUSION",size=13,bold=True)
for p in [
"This project implemented and verified a modular suicide-risk research-screening prototype using NLP-based text analysis and complementary speech distress analysis. The SDCNL text data were cleaned with explicit duplicate handling and leakage verification, producing a final 1,513-row training set and a fixed 379-row held-out test set with zero normalized text overlap.",
"The standard original-web-label comparison retained BiLSTM, frozen BERT and TF-IDF + calibrated Linear SVM, with the standard SVM reaching 72.56% accuracy. The enhanced SDCNL-style label-correction pipeline uses BERT embeddings, UMAP and GMM confidence correction at τ=0.90, followed by a fast word+character TF-IDF + calibrated LinearSVC. The resulting primary label-corrected evaluation is 85.75% accuracy, 85.25% F1 and 95.04% ROC-AUC. A separate offline corrected-label BERT experiment reached 93.14% accuracy. These corrected-label results are explicitly separated from the standard original-web-label baseline. The speech branch was implemented using EATD acoustic features and participant-grouped model selection.",
"The final Streamlit application supports text-only, speech-only and combined modes. Since SDCNL and EATD describe different participants and different targets, the project intentionally avoids invalid row-wise multimodal training. The combined decision is a transparent prototype late-fusion evidence score rather than a clinically calibrated suicide-risk probability.",
"The final implementation, models, diagrams, documentation and automated verification are complete for the available datasets and the scientifically supportable project architecture. The system is suitable for academic demonstration and research experimentation, but it is not a substitute for professional mental-health assessment or clinical diagnosis."
]: add_body(doc,p)

# Future
chapter_page(doc,10); add_center(doc,"10. FUTURE ENHANCEMENT",size=13,bold=True)
for p in [
"Future work should prioritize collection or access to a genuinely paired multimodal dataset in which speech, text and suicide-risk labels come from the same participants. Such data would enable supervised multimodal fusion and statistically valid estimation of final multimodal performance.",
"The speech branch can be strengthened using larger multilingual datasets, self-supervised speech representations and external validation across languages and demographic groups. Because EATD is primarily Chinese and uses depression/distress labels, future suicide-specific acoustic datasets would improve target validity.",
"The text branch can be extended with fully fine-tuned transformer models, stronger calibration, explainability methods and robust evaluation under domain shift. Any increase in model complexity should continue to use leakage-free evaluation and fixed held-out testing.",
"Finally, a future production research platform could add consent-aware data collection, secure storage, audit logging and clinician-reviewed escalation pathways. Such deployment would require ethical review, privacy controls and independent clinical validation before use in real-world healthcare."
]: add_body(doc,p)

# References
doc.add_page_break(); add_center(doc,"REFERENCES",size=14,bold=True)
refs=[
'[1] A. Haque, V. Reddi, and T. Giallanza, “Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction,” in Proc. 30th International Conference on Artificial Neural Networks (ICANN), 2021, pp. 436–447, doi: 10.1007/978-3-030-86383-8_35.',
'[2] Y. Shen, H. Yang, and L. Lin, “Automatic Depression Detection: an Emotional Audio-Textual Corpus and A GRU/BiLSTM-Based Model,” in Proc. IEEE ICASSP, 2022, doi: 10.1109/ICASSP43922.2022.9746569.',
'[3] J. Pokrywka, J. I. Kaczmarek, and E. J. Gorzelańczyk, “Evaluating Transformer Models for Suicide Risk Detection on Social Media,” in 2024 IEEE International Conference on Big Data, 2024, doi: 10.1109/BigData62323.2024.10826094.',
'[4] J. Jin, Y. Liu, and Y. Dai, “Multimodal Speech and Text Models to Detect Suicidal Risks in Adolescents,” medRxiv, 2025, doi: 10.1101/2025.07.31.25332367.',
'[5] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in NAACL-HLT, 2019.',
'[6] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.',
'[7] B. McFee et al., “librosa: Audio and Music Signal Analysis in Python,” in Proc. 14th Python in Science Conference, 2015.',
'[8] SDCNL project repository and dataset, A. Haque et al., https://github.com/ayaanzhaque/SDCNL.',
'[9] EATD-Corpus project repository, https://github.com/speechandlanguageprocessing/ICASSP2022-Depression.',
'[10] Streamlit documentation, https://docs.streamlit.io/.'
]
for x in refs:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(7)
    r=p.add_run(x);set_run_font(r,size=10.5)
doc.add_page_break()
add_body(doc,"Note: Reference metadata for the core SDCNL and EATD studies is based on their published proceedings and official project pages. Project-specific experimental results in this report are taken from the verified local model metric files.")

# Appendix / article
doc.add_page_break(); add_center(doc,"APPENDIX",size=15,bold=True)
appsec=doc.add_section(WD_SECTION.NEW_PAGE)
appsec.page_width=Inches(8.27); appsec.page_height=Inches(11.69); appsec.top_margin=Inches(.6); appsec.bottom_margin=Inches(.6); appsec.left_margin=Inches(.55); appsec.right_margin=Inches(.55)
add_page_border(appsec); set_columns(appsec,2,300); add_page_number(appsec,"decimal")
add_center(doc,TITLE,size=13,bold=True)
add_center(doc,f"{GUIDE} | {STUDENTS[0][0]} | {STUDENTS[1][0]} | {STUDENTS[2][0]}",size=8.5)
add_center(doc,"Department of Information Technology, Lakireddy Bali Reddy College of Engineering, Mylavaram, India",size=8)
for h,txt in [
("Abstract—","This article presents the current implementation of a multimodal suicide-risk research-screening framework using independent NLP text analysis and acoustic speech analysis. The SDCNL text branch compares BiLSTM, frozen BERT embeddings with Logistic Regression, and TF-IDF with calibrated Linear SVM. On the fixed leakage-free test set, the calibrated SVM achieved 72.56% accuracy, 73.74% F1 and 78.22% ROC-AUC. The EATD speech branch uses participant-safe acoustic modelling and provides a complementary depression/distress score. Because the datasets are unmatched, only transparent late decision fusion is used."),
("I. INTRODUCTION","Automated analysis of digital language and speech can support research into early mental-health screening. This project focuses on a scientifically cautious architecture that keeps text suicide-related evidence separate from speech depression/distress evidence until the final decision stage."),
("II. METHODOLOGY","The text branch uses leakage-free SDCNL splits and three model families. The speech branch uses EATD audio processed at 16 kHz with MFCC, pitch, RMS, ZCR, spectral centroid and duration features. Participant-grouped cross-validation is used for speech model selection. Combined mode applies a transparent 0.70/0.30 text/speech weighting."),
("III. RESULTS","The calibrated text SVM achieved 72.56% accuracy, 71.92% precision, 75.65% recall, 73.74% F1 and 78.22% ROC-AUC. Participant-aggregated EATD validation for the operational speech model achieved 63.29% accuracy, 63.44% balanced accuracy and 59.22% ROC-AUC. The final automated verification suite passed 56/56 checks."),
("IV. CONCLUSION","The project demonstrates a complete academic prototype with independent branch evaluation, transparent fusion, Streamlit UI and automated verification. The system is not clinically validated and does not claim a final multimodal accuracy because the underlying datasets are not paired.")
]:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(5)
    r=p.add_run(h+" ");set_run_font(r,size=8.7,bold=True); r=p.add_run(txt);set_run_font(r,size=8.7)
add_heading(doc,"REFERENCES",2)
for x in refs[:4]:
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after=Pt(3)
    r=p.add_run(x);set_run_font(r,size=7.8)

doc.save(DOCX_OUT)

# ----------------- PDF -----------------
# Font
for font_name,font_path in [("TNR","C:/Windows/Fonts/times.ttf"),("TNRB","C:/Windows/Fonts/timesbd.ttf"),("Consolas","C:/Windows/Fonts/consola.ttf")]:
    try: pdfmetrics.registerFont(TTFont(font_name,font_path))
    except: pass

styles=getSampleStyleSheet()
body=ParagraphStyle("Body",fontName="TNR",fontSize=10.8,leading=15,alignment=TA_JUSTIFY,spaceAfter=6)
head=ParagraphStyle("Head",fontName="TNRB",fontSize=12,leading=15,spaceBefore=6,spaceAfter=6)
cent=ParagraphStyle("Cent",fontName="TNR",fontSize=11,leading=14,alignment=TA_CENTER)
centb=ParagraphStyle("CentB",fontName="TNRB",fontSize=12,leading=15,alignment=TA_CENTER)
titleStyle=ParagraphStyle("Title",fontName="TNRB",fontSize=16,leading=20,alignment=TA_CENTER)
small=ParagraphStyle("Small",fontName="TNR",fontSize=9,leading=11,alignment=TA_JUSTIFY)
code_style=ParagraphStyle("Code",fontName="Consolas",fontSize=5.7,leading=7.0,leftIndent=4,rightIndent=4)

def footer(canvas,doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(colors.black); canvas.setLineWidth(.7)
    canvas.rect(7*mm,7*mm,w-14*mm,h-14*mm)
    canvas.setFont("TNR",8)
    canvas.drawCentredString(w/2,9*mm,str(doc.page))
    canvas.restoreState()

story=[]
def P(txt,sty=body): story.append(Paragraph(txt.replace("\n","<br/>"),sty))
def PB(): story.append(PageBreak())
def IMG(path,w=165*mm):
    if Path(path).exists():
        im=RLImage(str(path),width=w,height=w*0.58); story.append(im); story.append(Spacer(1,3*mm))
def CH(n):
    PB(); story.append(Spacer(1,80*mm)); P(f"CHAPTER - {n}",ParagraphStyle("Ch",parent=titleStyle,fontSize=15)); PB()

# Front matter PDF
story.append(Spacer(1,15*mm)); P(TITLE,titleStyle); P("A Project Report",cent); story.append(Spacer(1,8*mm))
P("Submitted in partial fulfilment of requirements for the award of the degree of",cent)
P("<b>BACHELOR OF TECHNOLOGY</b><br/><b>IN</b><br/><b>INFORMATION TECHNOLOGY</b>",cent)
story.append(Spacer(1,5*mm)); P("<b>Submitted by</b>",cent)
for n,rn in STUDENTS:P(f"<b>{n} ({rn})</b>",cent)
story.append(Spacer(1,4*mm));P("<b>Under the Esteemed Guidance of</b>",cent);P(f"<b>{GUIDE}</b><br/>{GUIDE_DESIG}<br/>Dept. of IT, LBRCE",cent)
story.append(Spacer(1,7*mm));P(f"<b>{DEPT}</b><br/><b>{COLLEGE}</b><br/>(AUTONOMOUS)<br/>An ISO 21001:2018, 14001:2015, 50001:2018 Certified Institution<br/>Approved by AICTE, New Delhi and Affiliated to JNTUK, Kakinada<br/>L.B. REDDY NAGAR, MYLAVARAM, NTR DIST., A.P.-521 230<br/><b>{YEAR}</b>",cent)
PB(); P(f"<b>{COLLEGE}</b><br/>(AUTONOMOUS)<br/>Accredited by NAAC & NBA (Under Tier - I)<br/>An ISO 21001:2018, 14001:2015, 50001:2018 Certified Institution<br/>Approved by AICTE, New Delhi and Affiliated to JNTUK, Kakinada<br/>L.B. REDDY NAGAR, MYLAVARAM, NTR DIST., A.P.-521 230",cent)
story.append(Spacer(1,8*mm));P(f"<b>{DEPT}</b>",cent);P("<b>CERTIFICATE</b>",titleStyle);P(cert)
story.append(Spacer(1,25*mm))
sig=Table([["Project Guide","Head of the Department"],[f"{GUIDE}\n{GUIDE_DESIG}","Dr. D. Ratna Kishore\nProfessor & HOD, IT"],["Project Co Ordinator","External Examiner"],["Mr. M. Vijay Kumar\nSr Assistant Professor",""]],colWidths=[85*mm,85*mm])
sig.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTSIZE",(0,0),(-1,-1),9),("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BOTTOMPADDING",(0,0),(-1,-1),8)]));story.append(sig)
PB();P("<b>ACKNOWLEDGEMENT</b>",titleStyle)
for x in acks:P(x)
PB();P("<b>CANDIDATE’S DECLARATION</b>",titleStyle);P(f"We hereby declare that the project report entitled “{TITLE}” is a record of our work carried out during the academic year {YEAR} under the guidance of {GUIDE}, {GUIDE_DESIG}, Department of Information Technology, and submitted in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Information Technology.");P("The matter embodied in this project is genuine work carried out by us and has not been submitted earlier to this university or any other university for the award of any degree, diploma or prize.")
story.append(Spacer(1,35*mm));P("<b>PROJECT ASSOCIATES</b>",cent);[P(f"<b>{n} ({rn})</b>",cent) for n,rn in STUDENTS]
PB();P("<b>INDEX</b>",titleStyle)
data=[["Contents","Page No"]]+[[a,b] for a,b in index_rows]
tab=Table(data,colWidths=[145*mm,25*mm],repeatRows=1);tab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.black),("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTNAME",(0,0),(-1,0),"TNRB"),("FONTSIZE",(0,0),(-1,-1),8.7),("ALIGN",(1,0),(1,-1),"CENTER")]))
story.append(tab)
PB();P("<b>LIST OF FIGURES</b>",titleStyle)
data=[["Figure No","Description","Page No"]]+[list(x) for x in figs]
tab=Table(data,colWidths=[25*mm,120*mm,25*mm],repeatRows=1);tab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.black),("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTNAME",(0,0),(-1,0),"TNRB"),("FONTSIZE",(0,0),(-1,-1),8.3),("ALIGN",(0,0),(-1,-1),"CENTER")]))
story.append(tab)
PB();P("<b>ACRONYMS AND ABBREVIATIONS</b>",titleStyle)
tab=Table(acrs,colWidths=[45*mm,125*mm]);tab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.black),("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTSIZE",(0,0),(-1,-1),9)]));story.append(tab)
PB();P("<b>ABSTRACT</b>",titleStyle);P(abstract)

# Body same sections, shorter but complete
CH(1);P("<b>1.1. INTRODUCTION:</b>",head)
for x in [
"Suicide and self-harm related risk is a major mental-health concern, and early identification of warning signals can support timely human intervention. Conventional assessment depends on professional interviews and validated clinical instruments. These remain essential for diagnosis, but they are difficult to scale continuously across large online populations.",
"Digital platforms have created new sources of behavioural and linguistic information. Natural Language Processing can model text patterns, while speech processing can capture complementary acoustic characteristics such as pitch variation, energy and spectral patterns.",
"This project studies a modular multimodal architecture with independent SDCNL text and EATD speech branches. Because the datasets contain different participants and different targets, only late decision-level fusion is used."
]:P(x)
PB();P("<b>1.2. EXISTING SYSTEM:</b>",head)
for x in ["Traditional clinical screening requires expert time and availability.","Automated text systems can be affected by noisy labels, leakage and calibration problems.","Public speech datasets with direct suicide-risk labels are scarce.","Unmatched datasets cannot be safely concatenated as if they were paired multimodal records."]:P("• "+x)
PB();P("<b>1.3. PROPOSED SYSTEM:</b>",head)
for x in ["Leakage-free SDCNL text pipeline.","BiLSTM, frozen BERT + LR, and calibrated TF-IDF SVM comparison.","EATD acoustic speech model with participant-grouped validation.","Late fusion of P_text and P_speech only.","Streamlit UI with Text, Speech and Both modes.","Automated 37-check final verification."]:P("• "+x)

CH(2);P("<b>2. LITERATURE SURVEY</b>",head)
for h,b in lit:P(f"<b>{h}:</b> {b}")
PB();P("<b>Research Gap</b>",head);P("Existing research shows the value of text, speech and multimodal analysis, but the public datasets available to this project are not participant-matched. The project therefore emphasizes leakage-free branch evaluation and transparent decision-level fusion rather than unsupported feature-level merging.")

CH(3);P("<b>3. SYSTEM SPECIFICATIONS</b>",head);P("<b>3.1 SOFTWARE REQUIREMENTS:</b>",head)
for a,b in soft:P(f"<b>{a}:</b> {b}",small)
PB();P("<b>3.2 HARDWARE REQUIREMENTS:</b>",head)
for x in ["Windows 10/11 64-bit","Intel Core i5/i7 or equivalent","8 GB RAM minimum; 16 GB preferred","10-20 GB free storage","GPU optional for final inference"]:P("• "+x)
PB();P("<b>3.3 FUNCTIONAL REQUIREMENTS:</b>",head)
for x in ["Text-only, speech-only and combined input","Saved-model inference","Audio feature extraction","Prototype fusion","Screening result display","Reusable artifacts without retraining"]:P("• "+x)
PB();P("<b>3.4 NON-FUNCTIONAL REQUIREMENTS:</b>",head)
for x in ["Usability","Reliability","Maintainability","Transparency","Reproducibility","Safe interpretation"]:P("• "+x)

CH(4);P("<b>4. METHODOLOGY</b>",head);P("<b>4.1 Overview of the Project</b>",head);P("The system contains independent text and speech branches followed by a transparent prototype decision layer.");IMG(ASSET/"figure_4_1_architecture.png");P("<b>Figure 4.1: System Architecture Diagram</b>",cent)
PB();P("<b>4.2 Text Data Pre-processing</b>",head);[P("• "+x) for x in ["clean model_text","remove empty inputs","remove exact duplicates","preserve labels","check normalized overlap","keep held-out test isolated"]];P("Final text split: 1,513 training rows and 379 held-out test rows with zero normalized overlap.")
PB();P("<b>4.3 Text Model Development</b>",head);[P("• "+x) for x in ["BiLSTM baseline","TF-IDF + calibrated Linear SVM","Frozen BERT embeddings + Logistic Regression","Same held-out test set for all comparisons"]];P("<b>4.4 Speech Pre-processing and Feature Extraction</b>",head);P("Audio is loaded at 16 kHz mono, peak normalized, and represented using MFCC, pitch, RMS, ZCR, spectral centroid and duration features.")
PB();P("<b>4.5 Speech Model Selection</b>",head);P("Participant-grouped cross-validation prevents participant leakage across folds. Official EATD validation is used only after selection.");P("<b>4.6 Prototype Late Fusion</b>",head);P("Combined = 0.70 × P_text + 0.30 × P_speech. Low < 0.40, Moderate < 0.70, High ≥ 0.70. These are prototype research settings, not clinical thresholds.")

CH(5);P("<b>5. DESIGN VIEW</b>",head)
for h,img,cap,desc in designs:
    P(f"<b>{h}</b>",head);P(desc);IMG(img,150*mm);P(f"<b>{cap}</b>",cent);PB()

CH(6);P("<b>6. IMPLEMENTATION</b>",head);P(impl_intro)
for i,(h,rel,nlines) in enumerate(files):
    P(f"<b>{h}</b>",head)
    pth=ROOT/rel; code=pth.read_text(encoding="utf-8",errors="replace") if pth.exists() else "# File not found"
    story.append(Preformatted("\n".join(code.splitlines()[:nlines]),code_style))
    if i<len(files)-1:PB()

CH(7);P("<b>7. TESTING</b>",head);P("The automated verification suite checks files, syntax, datasets, metrics, diagrams and branch inference.");P("<b>7.1 Types of Testing Implemented</b>",head)
for h,b in [("Unit Testing","individual functions"),("Integration Testing","model and UI interaction"),("Dataset Integrity Testing","row counts, labels and leakage"),("System Testing","complete app"),("Diagram Verification","six required diagrams")]:P(f"<b>{h}:</b> {b}.")
PB();P("<b>7.2 Test Cases and Results</b>",head)
data=[["ID","Test Description","Expected Output","Result"]]+[list(x) for x in tests]
tab=Table(data,colWidths=[18*mm,60*mm,75*mm,20*mm],repeatRows=1);tab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.black),("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTNAME",(0,0),(-1,0),"TNRB"),("FONTSIZE",(0,0),(-1,-1),8)]));story.append(tab)

CH(8);P("<b>8. RESULTS AND EVALUATION</b>",head);P("The standard original-web-label baseline is preserved, while the primary enhanced fast model uses SDCNL-style UMAP-GMM corrected labels and achieves 85.75% accuracy, 85.25% F1 and 95.04% ROC-AUC. A separate offline corrected-label BERT experiment reached 93.14% accuracy.");IMG(ASSET/"figure_8_1_text_model_comparison.png",150*mm);P("<b>Figure 8.1: Text Model Performance Comparison</b>",cent)
PB();P("<b>8.1 Text Results</b>",head)
data=[["Model","Accuracy","Precision","Recall","F1","ROC-AUC"],["BiLSTM","59.37","57.77","75.13","65.32","63.58"],["Frozen BERT + LR","64.12","63.77","68.39","66.00","72.27"],["TF-IDF + SVM (original labels)","72.56","71.92","75.65","73.74","78.22"],["Enhanced Fast (corrected labels)","85.75","93.98","78.00","85.25","95.04"]]
tab=Table(data,colWidths=[52*mm,25*mm,25*mm,22*mm,20*mm,26*mm]);tab.setStyle(TableStyle([("GRID",(0,0),(-1,-1),.5,colors.black),("FONTNAME",(0,0),(-1,-1),"TNR"),("FONTNAME",(0,0),(-1,0),"TNRB"),("FONTSIZE",(0,0),(-1,-1),8)]));story.append(tab);IMG(ASSET/"figure_8_2_text_cm.png",95*mm);P("<b>Standard original-label confusion matrix</b>",cent);IMG(ASSET/"figure_8_2b_enhanced_text_cm.png",95*mm);P("<b>Enhanced label-corrected confusion matrix</b>",cent)
PB();P("<b>8.2 Speech Results</b>",head);P("Participant-level EATD validation: accuracy 63.29%, balanced accuracy 63.44%, precision 21.88%, recall 63.64%, F1 32.56%, ROC-AUC 59.22%.");IMG(ASSET/"figure_8_3_speech_cm.png",105*mm)
PB();P("<b>8.3 Application Output</b>",head)
for img,cap in [(ROOT/"reports"/"OUTPUT_01_TEXT_ONLY.png","Text-only"),(ROOT/"reports"/"OUTPUT_02_SPEECH_ONLY.png","Speech-only"),(ROOT/"reports"/"OUTPUT_03_TEXT_AND_SPEECH.png","Text + Speech")]:
    IMG(img,150*mm);P(f"<b>{cap} output</b>",cent)
IMG(ASSET/"figure_8_7_test_summary.png",120*mm);P("<b>Final automated verification: all 56 implemented checks passed</b>",cent)

CH(9);P("<b>9. CONCLUSION</b>",head)
for p in ["The project implemented a complete modular research-screening prototype using NLP text analysis and complementary acoustic speech analysis.","The primary production text branch is a fast word+character TF-IDF + calibrated LinearSVC trained on SDCNL-style UMAP-GMM confidence-corrected labels, achieving 85.75% label-corrected accuracy. The standard 72.56% original-web-label SVM baseline is retained separately. The speech model is reported with modest validation performance and is not overstated.","The application supports all three requested input modes and passed the final automated test suite.","Because the datasets are unmatched, the combined score is a transparent prototype evidence score and not a clinically validated suicide-risk probability."]:P(p)

CH(10);P("<b>10. FUTURE ENHANCEMENT</b>",head)
for p in ["Collect or access paired multimodal suicide-labelled data.","Improve multilingual and suicide-specific speech modelling.","Evaluate fully fine-tuned transformers under leakage-free testing.","Add secure, consent-aware data management and independent clinical validation before any real healthcare deployment."]:P(p)

PB();P("<b>REFERENCES</b>",titleStyle)
for x in refs:P(x,small)
PB();P("<b>APPENDIX</b>",titleStyle);P(f"<b>{TITLE}</b>",titleStyle);P(f"{GUIDE} | {STUDENTS[0][0]} | {STUDENTS[1][0]} | {STUDENTS[2][0]}",cent);P("Department of Information Technology, Lakireddy Bali Reddy College of Engineering, Mylavaram, India",cent)
for h,txt in [("Abstract—","This article presents the verified current implementation of the project, including leakage-free SDCNL text modelling, EATD acoustic speech modelling and transparent prototype late fusion."),("I. INTRODUCTION","The aim is to support academic research into multimodal screening without overclaiming clinical validity."),("II. METHODOLOGY","Independent text and speech branches are evaluated separately and combined only at the final decision layer."),("III. RESULTS","The primary enhanced fast text model achieved 85.75% label-corrected accuracy and 95.04% ROC-AUC; the standard original-web-label SVM baseline remains 72.56%. A separate offline label-corrected BERT experiment reached 93.14% accuracy. The final automated verification suite passed all 56 implemented checks."),("IV. CONCLUSION","The implementation is complete as a research prototype; clinical multimodal validation remains outside the evidence supported by the unmatched datasets.")]:P(f"<b>{h}</b> {txt}",small)
P("<b>REFERENCES</b>",head)
for x in refs[:4]:P(x,small)

pdf=SimpleDocTemplate(str(PDF_OUT),pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm)
pdf.build(story,onFirstPage=footer,onLaterPages=footer)

print("DOCX",DOCX_OUT,DOCX_OUT.stat().st_size)
print("PDF",PDF_OUT,PDF_OUT.stat().st_size)
