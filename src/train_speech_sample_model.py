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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, balanced_accuracy_score, classification_report

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA=BASE/"Datasets"/"Processed_EATD"
MODEL_DIR=BASE/"models"/"speech_model"
REPORT_DIR=BASE/"reports"/"speech_model"
MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

train=pd.read_csv(DATA/"EATD-train-features.csv",encoding="utf-8-sig")
val=pd.read_csv(DATA/"EATD-validation-features.csv",encoding="utf-8-sig")
TARGET="depression_label"
GROUP="participant_id"
EXCLUDE={GROUP,"raw_sds_score","standard_sds_score",TARGET,"emotion","transcript","audio_file","audio_source_type",
         "original_duration_sec","original_sample_rate","original_channels","sample_rate_model"}
features=[c for c in train.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(train[c]) and train[c].nunique(dropna=True)>1]

X=train[features].copy(); y=train[TARGET].astype(int).values; groups=train[GROUP].astype(str).values
Xv=val[features].copy(); yv=val[TARGET].astype(int).values; gv=val[GROUP].astype(str).values

def binary_metrics(y_true,probs,thr):
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

def participant_aggregate(df, probs):
    tmp=pd.DataFrame({
      "participant_id":df[GROUP].astype(str).values,
      "true_label":df[TARGET].astype(int).values,
      "probability":probs
    })
    return tmp.groupby("participant_id",as_index=False).agg(
      true_label=("true_label","first"), probability=("probability","mean")
    )

cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
specs=[]
for k in [8,12,16,20,25,30,35]:
    if k>len(features): continue
    for C in [0.2,0.5,1.0]:
        specs.append((f"lr_k{k}_c{C}",Pipeline([
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler()),
          ("select",SelectKBest(f_classif,k=k)),
          ("model",LogisticRegression(max_iter=5000,class_weight="balanced",C=C,random_state=42))
        ])))
    for C in [0.5,1.0,2.0]:
        specs.append((f"rbf_k{k}_c{C}",Pipeline([
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler()),
          ("select",SelectKBest(f_classif,k=k)),
          ("model",SVC(C=C,gamma="scale",class_weight="balanced",probability=True,random_state=42))
        ])))

candidate_results={}
best=None
for name,pipe in specs:
    oof=cross_val_predict(pipe,X,y,cv=cv,groups=groups,method="predict_proba",n_jobs=1)[:,1]
    agg=participant_aggregate(train,oof)
    yy=agg["true_label"].values; pp=agg["probability"].values
    best_m=None
    for thr in np.linspace(0.15,0.75,121):
        m=binary_metrics(yy,pp,thr)
        score=(m["f1_score"],m["balanced_accuracy"],m["roc_auc"],m["recall"])
        if best_m is None or score>(best_m["f1_score"],best_m["balanced_accuracy"],best_m["roc_auc"],best_m["recall"]):
            best_m=m
    candidate_results[name]=best_m
    rank=(best_m["f1_score"],best_m["balanced_accuracy"],best_m["roc_auc"])
    if best is None or rank>best[0]:
        best=(rank,name,pipe,best_m["threshold"],best_m)

_,best_name,best_pipe,best_thr,best_oof=best
best_pipe.fit(X,y)
val_sample_probs=best_pipe.predict_proba(Xv)[:,1]
val_sample_metrics=binary_metrics(yv,val_sample_probs,best_thr)
val_part=participant_aggregate(val,val_sample_probs)
val_part_metrics=binary_metrics(val_part["true_label"].values,val_part["probability"].values,best_thr)
val_part_default=binary_metrics(val_part["true_label"].values,val_part["probability"].values,0.5)

joblib.dump(best_pipe,MODEL_DIR/"best_speech_single_audio_model.joblib")
selected=[features[i] for i in best_pipe.named_steps["select"].get_support(indices=True)]
config={
 "model_scope":"single audio acoustic depression/distress estimator",
 "best_model":best_name,"threshold":best_thr,
 "all_input_features":features,"selected_features":selected,
 "train_audio_rows":len(train),"validation_audio_rows":len(val),
 "train_participants":int(train[GROUP].nunique()),"validation_participants":int(val[GROUP].nunique()),
 "cv":"5-fold StratifiedGroupKFold by participant; no participant crosses train CV folds",
 "selection":"participant-aggregated OOF F1, then balanced accuracy, then ROC-AUC",
 "final_validation":"official EATD validation split only after selection",
 "important_limit":"EATD outcome is SDS-derived depression/distress; not direct suicide risk."
}
summary={
 "config":config,
 "training_grouped_oof_participant_metrics":best_oof,
 "official_validation_sample_level_metrics":val_sample_metrics,
 "official_validation_participant_aggregated_metrics":val_part_metrics,
 "official_validation_participant_default_0.5_metrics":val_part_default
}
with open(MODEL_DIR/"speech_single_audio_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2)
with open(REPORT_DIR/"speech_single_audio_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
with open(REPORT_DIR/"speech_single_audio_candidates.json","w",encoding="utf-8") as f: json.dump(candidate_results,f,indent=2)
pd.DataFrame({
 "participant_id":val[GROUP],"emotion":val["emotion"],"true_label":yv,
 "distress_probability":val_sample_probs,
 "predicted_label":(val_sample_probs>=best_thr).astype(int)
}).to_csv(REPORT_DIR/"speech_single_audio_validation_predictions.csv",index=False)
val_part.assign(predicted_label=(val_part["probability"]>=best_thr).astype(int)).to_csv(REPORT_DIR/"speech_participant_aggregated_validation_predictions.csv",index=False)
with open(REPORT_DIR/"speech_single_audio_classification_report.txt","w",encoding="utf-8") as f:
    f.write("SAMPLE LEVEL\n")
    f.write(classification_report(yv,(val_sample_probs>=best_thr).astype(int),digits=4,zero_division=0))
    f.write("\nPARTICIPANT AGGREGATED\n")
    f.write(classification_report(val_part["true_label"],(val_part["probability"]>=best_thr).astype(int),digits=4,zero_division=0))
print(json.dumps(summary,indent=2))
