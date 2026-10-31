
import json,joblib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,roc_auc_score,confusion_matrix,classification_report

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
D=BASE/"Datasets"; LC=BASE/"models"/"sdcnl_label_correction_2026-10-31"
MOD=BASE/"models"/"enhanced_text_model"; OUT=BASE/"reports"/"enhanced_text_model"
MOD.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
tr=pd.read_csv(D/"training-set-clean.csv"); te=pd.read_csv(D/"testing-set-clean-leakage-free.csv")
yc=np.load(LC/"corrected_train_labels.npy"); ytc=np.load(LC/"corrected_test_labels.npy")
orig=te.is_suicide.astype(int).values
texts=tr.model_text.fillna("").astype(str).values; ttexts=te.model_text.fillna("").astype(str).values

wv=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=.995,sublinear_tf=True,max_features=180000,dtype=np.float32)
cvect=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=180000,dtype=np.float32)
W=wv.fit_transform(texts); Cc=cvect.fit_transform(texts); TW=wv.transform(ttexts); TC=cvect.transform(ttexts)
CHAR_WEIGHT=0.7; C=0.6
X=hstack([W,Cc*CHAR_WEIGHT],format="csr"); T=hstack([TW,TC*CHAR_WEIGHT],format="csr")

cv=StratifiedKFold(5,shuffle=True,random_state=42)
base=LinearSVC(C=C,random_state=42,max_iter=10000)
oof=cross_val_predict(base,X,yc,cv=cv,method="decision_function",n_jobs=1)
def met(y,p,s):
    pr,re,f,_=precision_recall_fscore_support(y,p,average="binary",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"precision":float(pr),"recall":float(re),"f1":float(f),
            "roc_auc":float(roc_auc_score(y,s)),"confusion_matrix":confusion_matrix(y,p).tolist()}
best_t=0.;best_m=None
for th in np.quantile(oof,np.linspace(.20,.80,101)):
    m=met(yc,(oof>=th).astype(int),oof)
    rank=(m["accuracy"],m["f1"],m["roc_auc"])
    if best_m is None or rank>(best_m["accuracy"],best_m["f1"],best_m["roc_auc"]):
        best_m=m;best_t=float(th)

svm=LinearSVC(C=C,random_state=42,max_iter=10000).fit(X,yc)
cal=CalibratedClassifierCV(LinearSVC(C=C,random_state=42,max_iter=10000),cv=5,method="sigmoid").fit(X,yc)
score=svm.decision_function(T); pred=(score>=best_t).astype(int); prob=cal.predict_proba(T)[:,1]
corr=met(ytc,pred,prob); origm=met(orig,pred,prob)
meta={
 "model":"Enhanced fast word+char TF-IDF LinearSVC trained on SDCNL UMAP-GMM label-corrected labels",
 "label_correction":"BERT embeddings -> UMAP -> GMM, tau=0.90",
 "evaluation_basis":"SDCNL-style label-corrected test labels (separate from original web-label baseline)",
 "word_ngram":[1,2],"char_ngram":[3,5],"char_weight":CHAR_WEIGHT,"C":C,"decision_threshold":best_t,
 "selection_note":"Representation fixed to word(1,2)+char_wb(3,5). char_weight and C selected from corrected-train 5-fold OOF; test labels not used for hyperparameter selection.",
 "oof_metrics":best_m,
 "label_corrected_test_metrics":corr,
 "original_web_label_test_metrics":origm,
 "train_rows":len(tr),"test_rows":len(te)
}
joblib.dump(wv,MOD/"word_vectorizer.joblib");joblib.dump(cvect,MOD/"char_vectorizer.joblib")
joblib.dump(svm,MOD/"linear_svm.joblib");joblib.dump(cal,MOD/"calibrated_svm.joblib")
(MOD/"config.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
(OUT/"metrics.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
(OUT/"classification_report_corrected.txt").write_text(classification_report(ytc,pred,digits=4,zero_division=0),encoding="utf-8")
pd.DataFrame({"record_id":te.record_id,"original_label":orig,"corrected_label":ytc,"prediction":pred,"probability":prob,"decision_score":score}).to_csv(OUT/"test_predictions.csv",index=False)
print(json.dumps(meta,indent=2),flush=True)
