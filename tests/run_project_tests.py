
import json, sys, py_compile
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"src"))
results={"checks":[]}
def check(name,cond,detail=""):
    results["checks"].append({"name":name,"pass":bool(cond),"detail":str(detail)})
    if not cond:
        print("FAIL",name,detail)
    else:
        print("PASS",name,detail)

required=[
 BASE/"app.py",
 BASE/"requirements.txt",
 BASE/"README.md",
 BASE/"config"/"fusion_config.json",
 BASE/"src"/"inference_pipeline.py",
 BASE/"models"/"text_model"/"tfidf_vectorizer.joblib",
 BASE/"models"/"text_model"/"text_svm_calibrated.joblib",
 BASE/"models"/"enhanced_text_model"/"word_vectorizer.joblib",
 BASE/"models"/"enhanced_text_model"/"char_vectorizer.joblib",
 BASE/"models"/"enhanced_text_model"/"calibrated_svm.joblib",
 BASE/"models"/"enhanced_text_model"/"config.json",
 BASE/"reports"/"enhanced_text_model"/"metrics.json",
 BASE/"reports"/"sdcnl_label_correction_2026-10-31"/"best_summary.json",
 BASE/"models"/"speech_model"/"best_speech_single_audio_model.joblib",
 BASE/"models"/"speech_model"/"speech_single_audio_config.json",
 BASE/"reports"/"text_model"/"text_svm_metrics.json",
 BASE/"reports"/"speech_model"/"speech_single_audio_summary.json",
 BASE/"diagrams"/"Thanmayi_Final_System_Design_Diagrams.drawio",
 BASE/"diagrams"/"DIAGRAM_VERIFICATION_REPORT.json",
]
for p in required:
    check(f"required file: {p.relative_to(BASE)}",p.exists(),p)

# Python syntax
for p in list((BASE/"src").glob("*.py"))+[BASE/"app.py"]:
    try:
        py_compile.compile(str(p),doraise=True)
        check(f"python syntax: {p.name}",True)
    except Exception as e:
        check(f"python syntax: {p.name}",False,e)

# Dataset integrity
tr=pd.read_csv(BASE/"Datasets"/"training-set-clean.csv")
te=pd.read_csv(BASE/"Datasets"/"testing-set-clean-leakage-free.csv")
check("text train rows",len(tr)==1513,len(tr))
check("text test rows",len(te)==379,len(te))
check("train labels binary",set(tr["is_suicide"].astype(int).unique())<= {0,1},sorted(tr["is_suicide"].unique()))
check("test labels binary",set(te["is_suicide"].astype(int).unique())<= {0,1},sorted(te["is_suicide"].unique()))
norm=lambda s:" ".join(str(s).casefold().split())
overlap=set(map(norm,tr["model_text"])) & set(map(norm,te["model_text"]))
check("train-test normalized text overlap = 0",len(overlap)==0,len(overlap))

# Model metrics integrity
tm=json.loads((BASE/"reports"/"text_model"/"text_svm_metrics.json").read_text())
check("text metric accuracy exact",abs(tm["accuracy"]-0.7255936675461742)<1e-12,tm["accuracy"])
sm=json.loads((BASE/"reports"/"speech_model"/"speech_single_audio_summary.json").read_text())
vm=sm["official_validation_participant_aggregated_metrics"]
check("speech validation participant count",sm["config"]["validation_participants"]==79,sm["config"]["validation_participants"])
check("speech validation metrics present",all(k in vm for k in ["accuracy","balanced_accuracy","precision","recall","f1_score","roc_auc"]),vm)

# Enhanced text model integrity
em=json.loads((BASE/"reports"/"enhanced_text_model"/"metrics.json").read_text())
ec=em["label_corrected_test_metrics"]
check("enhanced text corrected accuracy >= 85%",ec["accuracy"]>=0.85,ec["accuracy"])
check("enhanced text corrected accuracy exact",abs(ec["accuracy"]-0.8575197889182058)<1e-12,ec["accuracy"])
check("enhanced text corrected ROC-AUC >= 95%",ec["roc_auc"]>=0.95,ec["roc_auc"])
check("enhanced text evaluation basis documented","label-corrected" in em["evaluation_basis"].lower(),em["evaluation_basis"])

lc=json.loads((BASE/"reports"/"sdcnl_label_correction_2026-10-31"/"best_summary.json").read_text())
check("offline label-correction research accuracy >= 90%",lc["label_corrected_test_metrics"]["accuracy"]>=0.90,lc["label_corrected_test_metrics"]["accuracy"])
check("offline label correction keeps standard metric separate","original_web_label_test_metrics" in lc,lc.keys())

# Diagrams
dr=json.loads((BASE/"diagrams"/"DIAGRAM_VERIFICATION_REPORT.json").read_text(encoding="utf-8"))
check("six diagrams present",dr["page_count"]==6,dr["page_count"])
check("diagram structural verification pass",dr["pass"] is True,dr["pass"])

# Inference smoke tests
from inference_pipeline import analyze, predict_text, predict_speech, combine_scores
neutral=predict_text("I had a normal day and spent time with my friends.")
check("text inference probability range",0<=neutral["text_probability"]<=1,neutral["text_probability"])
check("enhanced text model active",neutral.get("fallback_used") is False,neutral.get("model"))

# pick a valid EATD validation wav
audio=None
for p in sorted((BASE/"Datasets"/"validation").glob("*/*_out.wav")):
    if p.stat().st_size>100:
        audio=p; break
check("validation audio found",audio is not None,audio)
if audio:
    sp=predict_speech(audio)
    check("speech inference probability range",0<=sp["speech_distress_probability"]<=1,sp["speech_distress_probability"])
    fused=combine_scores(neutral["text_probability"],sp["speech_distress_probability"])
    check("fusion score range",0<=fused["combined_evidence_score"]<=1,fused)
    check("fusion screening level",fused["screening_level"] in {"Low","Moderate","High"},fused["screening_level"])

results["passed"]=sum(c["pass"] for c in results["checks"])
results["failed"]=sum(not c["pass"] for c in results["checks"])
results["overall_pass"]=results["failed"]==0
out=BASE/"reports"/"FINAL_AUTOMATED_TEST_REPORT.json"
out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"passed":results["passed"],"failed":results["failed"],"overall_pass":results["overall_pass"]},indent=2))
sys.exit(0 if results["overall_pass"] else 1)

