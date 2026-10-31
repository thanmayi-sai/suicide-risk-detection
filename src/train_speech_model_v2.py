import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, balanced_accuracy_score, classification_report

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA=BASE/"Datasets"/"Processed_EATD"
MODEL_DIR=BASE/"models"/"speech_model"
REPORT_DIR=BASE/"reports"/"speech_model"
MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

train=pd.read_csv(DATA/"EATD-train-participant-features.csv",encoding="utf-8-sig")
val=pd.read_csv(DATA/"EATD-validation-participant-features.csv",encoding="utf-8-sig")
TARGET="depression_label"
drop={"participant_id","raw_sds_score","standard_sds_score",TARGET}
features=[]
for c in train.columns:
    if c in drop: continue
    lc=c.lower()
    if "transcript" in lc or "source_type" in lc: continue
    if pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True)>1:
        features.append(c)
X=train[features].copy()
Xv=val[features].copy()
y=train[TARGET].astype(int).values
yv=val[TARGET].astype(int).values

cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
candidate_specs=[]
for k in [10,15,20,30,40,60,80,105]:
    if k>len(features): continue
    candidate_specs.append((f"lr_k{k}", Pipeline([
      ("imputer",SimpleImputer(strategy="median")),
      ("scaler",StandardScaler()),
      ("select",SelectKBest(f_classif,k=k)),
      ("model",LogisticRegression(max_iter=5000,class_weight="balanced",C=0.3,random_state=42))
    ])))
    candidate_specs.append((f"rbf_k{k}", Pipeline([
      ("imputer",SimpleImputer(strategy="median")),
      ("scaler",StandardScaler()),
      ("select",SelectKBest(f_classif,k=k)),
      ("model",SVC(C=0.7,gamma="scale",class_weight="balanced",probability=True,random_state=42))
    ])))

def metrics(y_true,probs,thr):
    pred=(probs>=thr).astype(int)
    p,r,f,_=precision_recall_fscore_support(y_true,pred,average="binary",zero_division=0)
    return {
      "threshold":float(thr),
      "accuracy":float(accuracy_score(y_true,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(y_true,pred)),
      "precision":float(p),"recall":float(r),"f1_score":float(f),
      "roc_auc":float(roc_auc_score(y_true,probs)),
      "confusion_matrix":confusion_matrix(y_true,pred).tolist()
    }

cv_results={}
best=None
for name,pipe in candidate_specs:
    oof=cross_val_predict(pipe,X,y,cv=cv,method="predict_proba",n_jobs=1)[:,1]
    best_thr=None; best_m=None
    for thr in np.linspace(0.15,0.75,121):
        m=metrics(y,oof,thr)
        score=(m["f1_score"],m["balanced_accuracy"],m["roc_auc"],m["recall"])
        if best_m is None or score>(best_m["f1_score"],best_m["balanced_accuracy"],best_m["roc_auc"],best_m["recall"]):
            best_thr=float(thr); best_m=m
    cv_results[name]=best_m
    rank=(best_m["f1_score"],best_m["balanced_accuracy"],best_m["roc_auc"])
    if best is None or rank>best[0]:
        best=(rank,name,pipe,best_thr,best_m)

_,best_name,best_pipe,best_thr,best_cv=best
best_pipe.fit(X,y)
val_probs=best_pipe.predict_proba(Xv)[:,1]
val_metrics=metrics(yv,val_probs,best_thr)
default_metrics=metrics(yv,val_probs,0.5)

# Bootstrap validation uncertainty (participant-level)
rng=np.random.default_rng(42)
boot=[]
n=len(yv)
for _ in range(2000):
    idx=rng.integers(0,n,n)
    yy=yv[idx]; pp=val_probs[idx]
    if len(np.unique(yy))<2: continue
    mm=metrics(yy,pp,best_thr)
    boot.append([mm["f1_score"],mm["balanced_accuracy"],mm["roc_auc"]])
arr=np.array(boot)
ci={}
for j,key in enumerate(["f1_score","balanced_accuracy","roc_auc"]):
    ci[key]=[float(np.percentile(arr[:,j],2.5)),float(np.percentile(arr[:,j],97.5))]

joblib.dump(best_pipe,MODEL_DIR/"best_speech_model_v2.joblib")
config={
 "best_model":best_name,"selected_threshold":best_thr,
 "feature_count_input":len(features),
 "selected_feature_names":[features[i] for i in best_pipe.named_steps["select"].get_support(indices=True)],
 "selection_method":"5-fold stratified cross-validation on training split only; choose model/k/threshold by OOF F1 then balanced accuracy then ROC-AUC",
 "validation_policy":"official EATD validation participants used only once after model/threshold selection",
 "train_participants":len(train),"validation_participants":len(val),
 "train_positive":int(y.sum()),"train_negative":int((y==0).sum()),
 "validation_positive":int(yv.sum()),"validation_negative":int((yv==0).sum()),
 "important_limit":"EATD predicts SDS-derived depression/distress, not direct suicide risk."
}
with open(MODEL_DIR/"speech_model_v2_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2,ensure_ascii=False)
summary={
 "best_model":best_name,
 "training_oof_metrics":best_cv,
 "official_validation_metrics_at_train_selected_threshold":val_metrics,
 "official_validation_metrics_at_default_0.5":default_metrics,
 "bootstrap_95pct_ci":ci,
 "config":config
}
with open(REPORT_DIR/"speech_model_v2_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
with open(REPORT_DIR/"speech_model_v2_cv_candidates.json","w",encoding="utf-8") as f: json.dump(cv_results,f,indent=2)
val_pred=(val_probs>=best_thr).astype(int)
pd.DataFrame({
 "participant_id":val["participant_id"],
 "true_label":yv,
 "predicted_label":val_pred,
 "distress_probability":val_probs
}).to_csv(REPORT_DIR/"speech_validation_predictions_v2.csv",index=False)
with open(REPORT_DIR/"speech_model_v2_classification_report.txt","w",encoding="utf-8") as f:
    f.write(classification_report(yv,val_pred,digits=4,zero_division=0))
print(json.dumps(summary,indent=2,ensure_ascii=False))
