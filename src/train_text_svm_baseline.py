import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA=BASE/"Datasets"
MODEL_DIR=BASE/"models"/"text_model"
REPORT_DIR=BASE/"reports"/"text_model"
train=pd.read_csv(DATA/"training-set-clean.csv").dropna(subset=["model_text","is_suicide"])
test=pd.read_csv(DATA/"testing-set-clean-leakage-free.csv").dropna(subset=["model_text","is_suicide"])

vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=0.98,max_features=50000,sublinear_tf=True)
Xtr=vec.fit_transform(train["model_text"].astype(str))
Xte=vec.transform(test["model_text"].astype(str))
base=LinearSVC(C=1.0,class_weight=None,random_state=42)
clf=CalibratedClassifierCV(base,cv=5,method="sigmoid")
clf.fit(Xtr,train["is_suicide"].astype(int))
pred=clf.predict(Xte)
probs=clf.predict_proba(Xte)[:,1]
y=test["is_suicide"].astype(int).values
acc=accuracy_score(y,pred)
p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
auc=roc_auc_score(y,probs)
cm=confusion_matrix(y,pred)
metrics={"model":"TF-IDF bigram + calibrated LinearSVC","accuracy":float(acc),"precision":float(p),"recall":float(r),"f1_score":float(f),"roc_auc":float(auc),"confusion_matrix":cm.tolist(),"train_rows":len(train),"test_rows":len(test)}
joblib.dump(vec,MODEL_DIR/"tfidf_vectorizer.joblib")
joblib.dump(clf,MODEL_DIR/"text_svm_calibrated.joblib")
with open(REPORT_DIR/"text_svm_metrics.json","w",encoding="utf-8") as fp: json.dump(metrics,fp,indent=2)
with open(REPORT_DIR/"text_svm_classification_report.txt","w",encoding="utf-8") as fp: fp.write(classification_report(y,pred,digits=4,zero_division=0))
print(json.dumps(metrics,indent=2))
