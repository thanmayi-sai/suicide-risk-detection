
import json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from umap import UMAP
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
D=BASE/"Datasets"; M=BASE/"models"/"text_model"
OUT=BASE/"reports"/"sdcnl_label_correction_2026-10-31"; MOD=BASE/"models"/"sdcnl_label_correction_2026-10-31"
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)

tr=pd.read_csv(D/"training-set-clean.csv"); te=pd.read_csv(D/"testing-set-clean-leakage-free.csv")
y=tr.is_suicide.astype(int).values; yt=te.is_suicide.astype(int).values
Etr=np.load(M/"bert_train_embeddings.npy").astype(np.float32)
Ete=np.load(M/"bert_test_embeddings.npy").astype(np.float32)
assert Etr.shape[0]==len(y) and Ete.shape[0]==len(yt)

# Standardize each BERT dimension before manifold reduction.
scaler=StandardScaler().fit(Etr)
Xtr=scaler.transform(Etr).astype(np.float32)
Xte=scaler.transform(Ete).astype(np.float32)
Xall=np.vstack([Xtr,Xte])
ntr=len(y)

configs=[
 {"n_neighbors":10,"n_components":2,"min_dist":0.0},
 {"n_neighbors":15,"n_components":2,"min_dist":0.0},
 {"n_neighbors":30,"n_components":2,"min_dist":0.0},
 {"n_neighbors":15,"n_components":5,"min_dist":0.0},
 {"n_neighbors":30,"n_components":5,"min_dist":0.0},
 {"n_neighbors":15,"n_components":10,"min_dist":0.0},
 {"n_neighbors":30,"n_components":10,"min_dist":0.0},
 {"n_neighbors":50,"n_components":10,"min_dist":0.0},
]
TAU=0.90
cv=StratifiedKFold(5,shuffle=True,random_state=42)

def metrics(yy,pred,prob):
    p,r,f,_=precision_recall_fscore_support(yy,pred,average="binary",zero_division=0)
    return {"accuracy":float(accuracy_score(yy,pred)),"precision":float(p),"recall":float(r),"f1":float(f),
            "roc_auc":float(roc_auc_score(yy,prob)),"confusion_matrix":confusion_matrix(yy,pred).tolist()}

results=[]; best=None
for cfg in configs:
    print("UMAP",cfg,flush=True)
    reducer=UMAP(n_neighbors=cfg["n_neighbors"],n_components=cfg["n_components"],min_dist=cfg["min_dist"],
                 metric="cosine",random_state=42,n_jobs=1)
    Z=reducer.fit_transform(Xall)
    Ztr,Zte=Z[:ntr],Z[ntr:]
    for cov in ["full","diag"]:
        g=GaussianMixture(n_components=2,covariance_type=cov,random_state=42,n_init=10,reg_covar=1e-5)
        g.fit(Ztr)  # fit GMM on training representation only; UMAP is transductive on features.
        train_cluster=g.predict(Ztr); probs_tr=g.predict_proba(Ztr)
        # map cluster ids to class labels using TRAIN labels only
        mapping={}
        for k in [0,1]:
            idx=np.where(train_cluster==k)[0]
            mapping[k]=int(np.bincount(y[idx],minlength=2).argmax()) if len(idx) else k
        # ensure two distinct mappings; if collapsed, use global assignment maximizing train agreement
        if len(set(mapping.values()))<2:
            acc_a=np.mean(np.array([0 if c==0 else 1 for c in train_cluster])==y)
            acc_b=np.mean(np.array([1 if c==0 else 0 for c in train_cluster])==y)
            mapping={0:0,1:1} if acc_a>=acc_b else {0:1,1:0}
        def corrected(Zpart,orig):
            cl=g.predict(Zpart); pp=g.predict_proba(Zpart); conf=pp.max(axis=1)
            mapped=np.array([mapping[int(c)] for c in cl],dtype=int)
            out=orig.copy()
            mask=conf>=TAU
            out[mask]=mapped[mask]
            return out,mask,conf,mapped,pp
        yc,mtr,ctr,maptr,ptr=corrected(Ztr,y)
        ytc,mte,cte,mapte,pte=corrected(Zte,yt)

        # classifier on original 768-d BERT embeddings, trained with corrected TRAIN labels
        clf=LogisticRegression(max_iter=5000,C=1.0,class_weight="balanced",random_state=42)
        oof=cross_val_predict(clf,Etr,yc,cv=cv,method="predict_proba",n_jobs=1)[:,1]
        oofm=metrics(yc,(oof>=.5).astype(int),oof)
        clf.fit(Etr,yc)
        prob=clf.predict_proba(Ete)[:,1]; pred=(prob>=.5).astype(int)
        corrected_m=metrics(ytc,pred,prob)
        original_m=metrics(yt,pred,prob)
        rec={
          "umap":cfg,"covariance":cov,"tau":TAU,"cluster_mapping":mapping,
          "train_corrected_count":int(mtr.sum()),"test_corrected_count":int(mte.sum()),
          "train_changed_count":int(np.sum(yc!=y)),"test_changed_count":int(np.sum(ytc!=yt)),
          "train_oof_corrected_label_metrics":oofm,
          "test_metrics_vs_label_corrected_ground_truth":corrected_m,
          "test_metrics_vs_original_web_labels":original_m
        }
        results.append(rec)
        print("cov",cov,"changed",rec["train_changed_count"],rec["test_changed_count"],
              "OOF",round(oofm["accuracy"],4),
              "CORR_TEST",round(corrected_m["accuracy"],4),round(corrected_m["f1"],4),round(corrected_m["roc_auc"],4),
              "ORIG_TEST",round(original_m["accuracy"],4),flush=True)
        rank=(oofm["accuracy"],oofm["f1"],oofm["roc_auc"])
        if best is None or rank>best[0]:
            best=(rank,cfg,cov,reducer,g,mapping,yc,ytc,mtr,mte,clf,rec,Z)

rank,cfg,cov,reducer,g,mapping,yc,ytc,mtr,mte,clf,rec,Z=best
# refit final classifier
clf=LogisticRegression(max_iter=5000,C=1.0,class_weight="balanced",random_state=42).fit(Etr,yc)
prob=clf.predict_proba(Ete)[:,1]; pred=(prob>=.5).astype(int)
final_corr=metrics(ytc,pred,prob); final_orig=metrics(yt,pred,prob)
summary={
 "method":"SDCNL-style unsupervised label correction: BERT embeddings -> UMAP -> GMM -> confidence threshold",
 "threshold_tau":TAU,
 "important_evaluation_note":"Label-corrected test metrics use corrected web-scraped test labels and are a separate SDCNL-style evaluation. They do NOT replace the standard held-out metrics against original labels.",
 "selection_rule":"UMAP/GMM config selected only by 5-fold OOF accuracy on corrected TRAIN labels.",
 "selected_umap":cfg,"selected_covariance":cov,"cluster_mapping":mapping,
 "train_label_changes":int(np.sum(yc!=y)),"test_label_changes":int(np.sum(ytc!=yt)),
 "label_corrected_test_metrics":final_corr,
 "original_web_label_test_metrics":final_orig
}
joblib.dump(scaler,MOD/"embedding_scaler.joblib"); joblib.dump(reducer,MOD/"umap_reducer.joblib"); joblib.dump(g,MOD/"gmm.joblib"); joblib.dump(clf,MOD/"bert_logreg_corrected.joblib")
np.save(MOD/"corrected_train_labels.npy",yc); np.save(MOD/"corrected_test_labels.npy",ytc)
(OUT/"all_configs.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
(OUT/"best_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
pd.DataFrame({"record_id":te.record_id,"original_label":yt,"corrected_label":ytc,"prediction":pred,"probability":prob}).to_csv(OUT/"test_predictions.csv",index=False)
print("FINAL",json.dumps(summary,indent=2),flush=True)
