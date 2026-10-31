from pathlib import Path
import pandas as pd, numpy as np, wave, re
BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000\Datasets")
OUT=BASE/"Processed_EATD"
EMOTIONS=["negative","neutral","positive"]

def rf(p):
    try: return float(p.read_text(encoding="utf-8-sig").strip())
    except: return np.nan

def validwav(p):
    if not p.exists(): return False
    try:
        with wave.open(str(p),"rb") as w:
            return w.getnframes()>0 and w.getframerate()>0
    except: return False

rows=[]
for split in ["train","validation"]:
    root=BASE/split
    for d in sorted([x for x in root.iterdir() if x.is_dir()], key=lambda p:int(p.name)):
        files=list(d.iterdir())
        raw=rf(d/"label.txt"); std=rf(d/"new_label.txt")
        r={"split":split,"participant_id":d.name,"folder_empty":len(files)==0,
           "raw_sds_score":raw,"standard_sds_score":std,
           "depression_label":int(std>=53) if not np.isnan(std) else np.nan,
           "raw_label_present":(d/"label.txt").exists(),"standard_label_present":(d/"new_label.txt").exists()}
        for e in EMOTIONS:
            r[f"{e}_transcript_present"]=(d/f"{e}.txt").exists() and bool((d/f"{e}.txt").read_text(encoding="utf-8-sig").strip())
            out=d/f"{e}_out.wav"; raww=d/f"{e}.wav"
            r[f"{e}_out_valid"]=validwav(out); r[f"{e}_raw_valid"]=validwav(raww)
            r[f"{e}_selected_source"]="preprocessed_out" if validwav(out) else ("raw_fallback" if validwav(raww) else "unusable")
        rows.append(r)
pd.DataFrame(rows).to_csv(OUT/"EATD-participant-manifest.csv",index=False,encoding="utf-8-sig")
print(pd.DataFrame(rows).groupby("split").size().to_dict())
print("fallbacks",[(r["split"],r["participant_id"],e) for r in rows for e in EMOTIONS if r[f"{e}_selected_source"]=="raw_fallback"])
print("empty",[(r["split"],r["participant_id"]) for r in rows if r["folder_empty"]])
print("missing raw label",[(r["split"],r["participant_id"]) for r in rows if (not r["folder_empty"]) and not r["raw_label_present"]])
