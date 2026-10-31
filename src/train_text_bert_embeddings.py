import json, joblib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
DATA=BASE/"Datasets"
MODEL_DIR=BASE/"models"/"text_model"
REPORT_DIR=BASE/"reports"/"text_model"
CACHE=Path(r"C:\Users\RAMESH\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")
MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

train=pd.read_csv(DATA/"training-set-clean.csv").dropna(subset=["model_text","is_suicide"]).copy()
test=pd.read_csv(DATA/"testing-set-clean-leakage-free.csv").dropna(subset=["model_text","is_suicide"]).copy()
device=torch.device("cpu")
tokenizer=AutoTokenizer.from_pretrained(str(CACHE),local_files_only=True)
encoder=AutoModel.from_pretrained(str(CACHE),local_files_only=True).to(device)
encoder.eval()

def embed_texts(texts,batch_size=16,max_len=128):
    out=[]
    texts=list(map(str,texts))
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        enc=tokenizer(batch,padding=True,truncation=True,max_length=max_len,return_tensors="pt")
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            res=encoder(**enc).last_hidden_state
            mask=enc["attention_mask"].unsqueeze(-1).float()
            pooled=(res*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy())
        print(f"embedded {min(i+batch_size,len(texts))}/{len(texts)}",flush=True)
    return np.vstack(out)

Xtr=embed_texts(train["model_text"].values)
Xte=embed_texts(test["model_text"].values)

np.save(MODEL_DIR/"bert_train_embeddings.npy",Xtr)
np.save(MODEL_DIR/"bert_test_embeddings.npy",Xte)

clf=LogisticRegression(max_iter=3000,C=1.0,random_state=42)
clf.fit(Xtr,train["is_suicide"].astype(int).values)

y=test["is_suicide"].astype(int).values
pred=clf.predict(Xte)
probs=clf.predict_proba(Xte)[:,1]
acc=accuracy_score(y,pred)
p,r,f,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
auc=roc_auc_score(y,probs)
cm=confusion_matrix(y,pred)

metrics={
 "model":"bert-base-uncased mean-pooled frozen embeddings + LogisticRegression",
 "embedding_model":"bert-base-uncased",
 "max_length":128,
 "train_rows":int(len(train)),
 "test_rows":int(len(test)),
 "accuracy":float(acc),"precision":float(p),"recall":float(r),"f1_score":float(f),"roc_auc":float(auc),
 "confusion_matrix":cm.tolist()
}
joblib.dump(clf,MODEL_DIR/"bert_embedding_logreg.joblib")
with open(MODEL_DIR/"bert_embedding_config.json","w",encoding="utf-8") as fp:
    json.dump({"local_model_path":str(CACHE),"pooling":"attention-mask mean pooling","classifier":"LogisticRegression","test_policy":"held-out leakage-free test only"},fp,indent=2)
with open(REPORT_DIR/"bert_embedding_metrics.json","w",encoding="utf-8") as fp: json.dump(metrics,fp,indent=2)
with open(REPORT_DIR/"bert_embedding_classification_report.txt","w",encoding="utf-8") as fp: fp.write(classification_report(y,pred,digits=4,zero_division=0))
print(json.dumps(metrics,indent=2),flush=True)
