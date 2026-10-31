import os, json, hashlib
from pathlib import Path
from collections import defaultdict
BASE=Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000")
report={"root":str(BASE),"files":0,"dirs":0,"zero_byte_files":[],"duplicate_basenames":{},"exact_duplicates_small":[],"key_deliverables":{}}
hashes=defaultdict(list); basenames=defaultdict(list)
for p in BASE.rglob("*"):
    if p.is_dir():
        report["dirs"]+=1; continue
    report["files"]+=1
    rel=str(p.relative_to(BASE))
    basenames[p.name.lower()].append(rel)
    try: size=p.stat().st_size
    except: continue
    if size==0: report["zero_byte_files"].append(rel)
    if size<=10*1024*1024:
        try:
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[(size,h)].append(rel)
        except: pass
for k,v in basenames.items():
    if len(v)>1: report["duplicate_basenames"][k]=v
for (size,h),v in hashes.items():
    if len(v)>1:
        report["exact_duplicates_small"].append({"size":size,"sha256":h,"files":v})
keys=[
 "app.py","README.md","requirements.txt","run_app.bat",
 "config/fusion_config.json",
 "src/inference_pipeline.py",
 "models/text_model/text_svm_calibrated.joblib",
 "models/speech_model/best_speech_single_audio_model.joblib",
 "reports/FINAL_AUTOMATED_TEST_REPORT.json",
 "diagrams/Thanmayi_Final_System_Design_Diagrams.drawio"
]
for k in keys: report["key_deliverables"][k]=(BASE/k).exists()
out=BASE/"reports"/"PROJECT_FILE_AUDIT_2026-10-29.json"
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
lines=[f"Root: {BASE}",f"Files: {report['files']}",f"Directories: {report['dirs']}",f"Zero-byte files: {len(report['zero_byte_files'])}",f"Duplicate basenames: {len(report['duplicate_basenames'])}",f"Exact duplicate groups (<=10MB): {len(report['exact_duplicates_small'])}","", "Key deliverables:"]
for k,v in report["key_deliverables"].items(): lines.append(f"- {k}: {'PASS' if v else 'MISSING'}")
lines+=["","Duplicate basenames (historical/versioned files are not automatically deleted):"]
for k,v in report["duplicate_basenames"].items(): lines.append(f"- {k}: {v}")
(BASE/"reports"/"PROJECT_FILE_AUDIT_2026-10-29.txt").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps({"files":report["files"],"dirs":report["dirs"],"zero":len(report["zero_byte_files"]),"duplicate_basenames":len(report["duplicate_basenames"]),"exact_duplicate_groups":len(report["exact_duplicates_small"]),"key_all":all(report["key_deliverables"].values())},indent=2))
