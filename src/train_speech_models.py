import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, balanced_accuracy_score, classification_report

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA=BASE/"Datasets"/"Processed_EATD"
MODEL_DIR=BASE/"models"/"speech_model"
REPORT_DIR=BASE/"reports"/"speech_model"
MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

TRAIN=DATA/"EATD-train-participant-features.csv"
VAL=DATA/"EATD-validation-participant-features.csv"
train=pd.read_csv(TRAIN,encoding="utf-8-sig")
val=pd.read_csv(VAL,encoding="utf-8-sig")

TARGET="depression_label"
DROP_EXACT={"participant_id","raw_sds_score","standard_sds_score",TARGET}
feature_cols=[]
for c in train.columns:
    if c in DROP_EXACT: continue
    lc=c.lower()
    if "transcript" in lc or "source_type" in lc: continue
    if pd.api.types.is_numeric_dtype(train[c]):
        feature_cols.append(c)

# remove constant columns (e.g., sample rate) based on train only
feature_cols=[c for c in feature_cols if train[c].nunique(dropna=True)>1]
Xtr=train[feature_cols].copy()
Xv=val[feature_cols].copy()
ytr=train[TARGET].astype(int).values
yv=val[TARGET].astype(int).values

models={
 "logistic_regression": Pipeline([
   ("imputer",SimpleImputer(strategy="median")),
   ("scaler",StandardScaler()),
   ("model",LogisticRegression(max_iter=5000,class_weight="balanced",C=1.0,random_state=42))
 ]),
 "rbf_svm": Pipeline([
   ("imputer",SimpleImputer(strategy="median")),
   ("scaler",StandardScaler()),
   ("model",SVC(kernel="rbf",C=1.0,gamma="scale",class_weight="balanced",probability=True,random_state=42))
 ]),
 "calibrated_linear_svm": Pipeline([
   ("imputer",SimpleImputer(strategy="median")),
   ("scaler",StandardScaler()),
   ("model",CalibratedClassifierCV(LinearSVC(C=1.0,class_weight="balanced",random_state=42),cv=3,method="sigmoid"))
 ]),
 "random_forest": Pipeline([
   ("imputer",SimpleImputer(strategy="median")),
   ("model",RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",min_samples_leaf=2,max_features="sqrt",random_state=42))
 ]),
 "gradient_boosting": Pipeline([
   ("imputer",SimpleImputer(strategy="median")),
   ("model",GradientBoostingClassifier(n_estimators=150,learning_rate=0.03,max_depth=2,random_state=42))
 ])
}

results={}
pred_rows=[]
for name,model in models.items():
    model.fit(Xtr,ytr)
    pred=model.predict(Xv)
    if hasattr(model,"predict_proba"):
        probs=model.predict_proba(Xv)[:,1]
    else:
        scores=model.decision_function(Xv)
        probs=1/(1+np.exp(-scores))
    acc=accuracy_score(yv,pred)
    bal=balanced_accuracy_score(yv,pred)
    p,r,f,_=precision_recall_fscore_support(yv,pred,average="binary",zero_division=0)
    auc=roc_auc_score(yv,probs)
    cm=confusion_matrix(yv,pred).tolist()
    results[name]={
      "accuracy":float(acc),"balanced_accuracy":float(bal),
      "precision":float(p),"recall":float(r),"f1_score":float(f),
      "roc_auc":float(auc),"confusion_matrix":cm
    }
    joblib.dump(model,MODEL_DIR/f"{name}.joblib")
    for pid,yt,yp,pr in zip(val["participant_id"],yv,pred,probs):
        pred_rows.append({"model":name,"participant_id":pid,"true_label":int(yt),"predicted_label":int(yp),"distress_probability":float(pr)})

# Choose by F1 first, then ROC-AUC, then balanced accuracy
best_name=max(results,key=lambda n:(results[n]["f1_score"],results[n]["roc_auc"],results[n]["balanced_accuracy"]))
best_model=models[best_name]
joblib.dump(best_model,MODEL_DIR/"best_speech_model.joblib")
config={
 "train_file":str(TRAIN),"validation_file":str(VAL),
 "target":TARGET,"feature_count":len(feature_cols),"features":feature_cols,
 "train_participants":int(len(train)),"validation_participants":int(len(val)),
 "train_positive":int(ytr.sum()),"train_negative":int((ytr==0).sum()),
 "validation_positive":int(yv.sum()),"validation_negative":int((yv==0).sum()),
 "selection_rule":"highest validation F1; ties by ROC-AUC then balanced accuracy",
 "best_model":best_name,
 "important_limit":"EATD depression/distress labels are not direct suicide-risk labels."
}
with open(MODEL_DIR/"speech_model_config.json","w",encoding="utf-8") as f: json.dump(config,f,indent=2)
with open(REPORT_DIR/"speech_model_comparison.json","w",encoding="utf-8") as f: json.dump(results,f,indent=2)
with open(REPORT_DIR/"speech_model_summary.json","w",encoding="utf-8") as f: json.dump({"best_model":best_name,"best_metrics":results[best_name],**config},f,indent=2)
pd.DataFrame(pred_rows).to_csv(REPORT_DIR/"speech_validation_predictions.csv",index=False)
with open(REPORT_DIR/"best_speech_classification_report.txt","w",encoding="utf-8") as f:
    pred=best_model.predict(Xv)
    f.write(classification_report(yv,pred,digits=4,zero_division=0))

print(json.dumps({"best_model":best_name,"best_metrics":results[best_name],"all_results":results,"feature_count":len(feature_cols)},indent=2))
