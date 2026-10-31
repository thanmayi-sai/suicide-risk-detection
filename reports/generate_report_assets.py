
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle, Ellipse
import numpy as np

ROOT=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
OUT=ROOT/"reports"/"final_report_assets"
OUT.mkdir(parents=True,exist_ok=True)

def save(fig,name):
    fig.savefig(OUT/name,dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def box(ax,xy,w,h,text,fs=10):
    x,y=xy
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.02",
                     linewidth=1.4,edgecolor="black",facecolor="white")
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,wrap=True)
    return p
def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=14,linewidth=1.2,color="black"))

# 4.1 Architecture
fig,ax=plt.subplots(figsize=(10,6)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
box(ax,(.04,.72),.18,.14,"User Interface\nText / Speech / Both")
box(ax,(.30,.78),.18,.12,"Text Branch\nSDCNL")
box(ax,(.30,.56),.18,.12,"Speech Branch\nEATD-Corpus")
box(ax,(.56,.78),.18,.12,"TF-IDF +\nCalibrated Linear SVM")
box(ax,(.56,.56),.18,.12,"Acoustic Features +\nLogistic Regression")
box(ax,(.79,.78),.16,.12,"P_text")
box(ax,(.79,.56),.16,.12,"P_speech")
box(ax,(.55,.30),.25,.13,"Prototype Late Fusion\n0.70 Text + 0.30 Speech")
box(ax,(.55,.08),.25,.13,"Low / Moderate / High\nResearch Screening Level")
arrow(ax,(.22,.79),(.30,.84)); arrow(ax,(.22,.77),(.30,.62))
arrow(ax,(.48,.84),(.56,.84)); arrow(ax,(.48,.62),(.56,.62))
arrow(ax,(.74,.84),(.79,.84)); arrow(ax,(.74,.62),(.79,.62))
arrow(ax,(.87,.78),(.69,.43)); arrow(ax,(.87,.56),(.69,.43))
arrow(ax,(.675,.30),(.675,.21))
ax.set_title("System Architecture Diagram",fontsize=16,fontweight="bold",pad=12)
save(fig,"figure_4_1_architecture.png")

# 5.1 Flow
fig,ax=plt.subplots(figsize=(7,9)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ys=[.90,.78,.65,.51,.37,.23,.09]
texts=["Start","Select Input Mode","Text / Speech / Both Input","Run Independent Branch Inference",
       "Generate P_text and/or P_speech","Apply Prototype Decision Layer","Display Result + Disclaimer"]
for y,t in zip(ys,texts):
    box(ax,(.23,y-.045),.54,.09,t,10)
for i in range(len(ys)-1): arrow(ax,(.5,ys[i]-.045),(.5,ys[i+1]+.045))
ax.set_title("System Flow Chart",fontsize=16,fontweight="bold")
save(fig,"figure_5_1_flow.png")

# 5.2 Logical
fig,ax=plt.subplots(figsize=(10,6)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
cols=[.04,.28,.52,.76]
labels=[
("Input Layer","Text Input\nSpeech Input"),
("Processing Layer","Text Cleaning / TF-IDF\nAudio 16 kHz + Features"),
("Model Layer","Calibrated Linear SVM\nSpeech Logistic Regression"),
("Decision Layer","P_text / P_speech\nLate Fusion\nScreening Band")]
for x,(h,b) in zip(cols,labels):
    box(ax,(x,.35),.19,.34,h+"\n\n"+b,10)
for i in range(3): arrow(ax,(cols[i]+.19,.52),(cols[i+1],.52))
ax.set_title("Logical Diagram",fontsize=16,fontweight="bold")
save(fig,"figure_5_2_logical.png")

# 5.3 Activity
fig,ax=plt.subplots(figsize=(7,9)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.add_patch(Circle((.5,.94),.018,color="black"))
steps=[(.84,"Choose Input Mode"),(.71,"Provide Required Input"),(.58,"Validate Input"),
       (.45,"Run Text/Speech Model"),(.32,"Combine Available Scores"),(.19,"Show Screening Result")]
for y,t in steps: box(ax,(.25,y-.045),.50,.09,t,10)
for a,b in zip([.922,.795,.665,.535,.405,.275],[.885,.755,.625,.495,.365,.235]): arrow(ax,(.5,a),(.5,b))
ax.add_patch(Circle((.5,.09),.027,fill=False,lw=1.5)); ax.add_patch(Circle((.5,.09),.014,color="black"))
arrow(ax,(.5,.145),(.5,.117))
ax.set_title("Behavioural / Activity Diagram",fontsize=16,fontweight="bold")
save(fig,"figure_5_3_activity.png")

# 5.4 ER conceptual
fig,ax=plt.subplots(figsize=(10,6)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
entities=[(.05,.62,"USER INPUT","input_id\ntext\naudio_path\nmode"),
          (.39,.62,"TEXT RESULT","text_probability\nmodel_name"),
          (.72,.62,"SPEECH RESULT","speech_probability\nthreshold"),
          (.39,.18,"SCREENING RESULT","combined_score\nscreening_level\nweights")]
for x,y,h,b in entities: box(ax,(x,y),.23,.22,h+"\n"+b,9)
arrow(ax,(.28,.73),(.39,.73)); arrow(ax,(.28,.67),(.72,.67))
arrow(ax,(.50,.62),(.50,.40)); arrow(ax,(.83,.62),(.58,.40))
ax.set_title("Conceptual ER Diagram",fontsize=16,fontweight="bold")
save(fig,"figure_5_4_er.png")

# 5.5 DFD
fig,ax=plt.subplots(figsize=(10,6)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
box(ax,(.03,.40),.18,.16,"User")
ax.add_patch(Ellipse((.42,.70),.28,.16,fill=False,lw=1.5)); ax.text(.42,.70,"1.0 Text\nAnalysis",ha="center",va="center")
ax.add_patch(Ellipse((.42,.35),.28,.16,fill=False,lw=1.5)); ax.text(.42,.35,"2.0 Speech\nAnalysis",ha="center",va="center")
ax.add_patch(Ellipse((.75,.52),.28,.18,fill=False,lw=1.5)); ax.text(.75,.52,"3.0 Decision /\nFusion",ha="center",va="center")
box(ax,(.79,.16),.17,.14,"Result Display")
arrow(ax,(.21,.50),(.29,.68)); arrow(ax,(.21,.48),(.29,.37))
arrow(ax,(.56,.70),(.65,.57)); arrow(ax,(.56,.35),(.65,.47))
arrow(ax,(.75,.43),(.86,.30))
ax.set_title("Data Flow Diagram - Level 1",fontsize=16,fontweight="bold")
save(fig,"figure_5_5_dfd.png")

# Text model comparison
models=["BiLSTM","Frozen BERT + LR","TF-IDF + Calibrated SVM"]
acc=[59.37,64.12,72.56]; f1=[65.32,66.00,73.74]; auc=[63.58,72.27,78.22]
x=np.arange(len(models)); w=.24
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(x-w,acc,w,label="Accuracy"); ax.bar(x,f1,w,label="F1"); ax.bar(x+w,auc,w,label="ROC-AUC")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Percent"); ax.set_ylim(0,90); ax.legend()
ax.set_title("Text Model Performance Comparison"); ax.grid(axis="y",alpha=.25)
save(fig,"figure_8_1_text_model_comparison.png")

# confusion helper
def cm_plot(cm,labels,title,name):
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    im=ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=14)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
    save(fig,name)
cm_plot([[129,57],[47,146]],["Class 0","Class 1"],"Text SVM Confusion Matrix","figure_8_2_text_cm.png")
cm_plot([[43,25],[4,7]],["Non-depressed","Depressed"],"Speech Model Confusion Matrix","figure_8_3_speech_cm.png")

# test summary
fig,ax=plt.subplots(figsize=(7,4)); ax.bar(["Passed","Failed"],[37,0]); ax.set_ylim(0,40); ax.set_title("Final Automated Test Summary")
for i,v in enumerate([37,0]): ax.text(i,v+1,str(v),ha="center",fontweight="bold")
save(fig,"figure_8_7_test_summary.png")
print(OUT)
