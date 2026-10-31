import os, re, wave, json
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

BASE = Path(r"C:\Users\RAMESH\Desktop\Projects\Thanmayi 8000\Datasets")
OUT = BASE / "Processed_EATD"
OUT.mkdir(exist_ok=True)
EMOTIONS = ["negative", "neutral", "positive"]
CUTOFF = 53.0

def clean_text(s):
    return re.sub(r"\s+", " ", s).strip()

def read_text(path):
    return clean_text(path.read_text(encoding="utf-8-sig")) if path.exists() else ""

def read_float(path):
    if not path.exists():
        return np.nan
    try:
        return float(path.read_text(encoding="utf-8-sig").strip())
    except:
        return np.nan

def wav_meta(path):
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            frames = w.getnframes()
            ch = w.getnchannels()
            dur = frames / sr if sr else 0
            return dur, sr, ch
    except:
        return 0.0, 0, 0

def choose_audio(pdir, emotion):
    outwav = pdir / f"{emotion}_out.wav"
    rawwav = pdir / f"{emotion}.wav"
    d, sr, ch = wav_meta(outwav)
    if outwav.exists() and d > 0:
        return outwav, "preprocessed_out", d, sr, ch
    d2, sr2, ch2 = wav_meta(rawwav)
    if rawwav.exists() and d2 > 0:
        return rawwav, "raw_fallback", d2, sr2, ch2
    return None, "missing_or_zero", 0.0, 0, 0

def extract_features(path):
    y, sr = librosa.load(path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    mx = np.max(np.abs(y))
    if mx > 0:
        y = y / mx
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    pitch = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    pitch = pitch[np.isfinite(pitch)]
    f = {
        "duration_sec": float(len(y) / sr),
        "sample_rate_model": int(sr),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "pitch_mean": float(np.mean(pitch)) if pitch.size else np.nan,
        "pitch_std": float(np.std(pitch)) if pitch.size else np.nan,
    }
    for i in range(13):
        f[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        f[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))
    return f
def process_split(split):
    root = BASE / split
    rows = []
    participant_rows = []
    anomalies = []
    pids = sorted([p.name for p in root.iterdir() if p.is_dir()], key=lambda x:int(x) if x.isdigit() else x)
    for pid in pids:
        pdir = root / pid
        files = set(x.name for x in pdir.iterdir() if x.is_file())
        if not files:
            anomalies.append({"participant_id":pid,"issue":"empty participant folder"})
            continue
        raw_score = read_float(pdir / "label.txt")
        standard_score = read_float(pdir / "new_label.txt")
        if np.isnan(standard_score):
            anomalies.append({"participant_id":pid,"issue":"missing/invalid new_label.txt"})
            continue
        depression_label = int(standard_score >= CUTOFF)
        participant_base = {
            "participant_id": pid,
            "raw_sds_score": raw_score,
            "standard_sds_score": standard_score,
            "depression_label": depression_label,
        }
        agg = dict(participant_base)
        valid_emotions = 0
        for emotion in EMOTIONS:
            transcript = read_text(pdir / f"{emotion}.txt")
            audio_path, source_type, orig_dur, orig_sr, orig_ch = choose_audio(pdir, emotion)
            if not transcript:
                anomalies.append({"participant_id":pid,"emotion":emotion,"issue":"missing/empty transcript"})
            if audio_path is None:
                anomalies.append({"participant_id":pid,"emotion":emotion,"issue":"no usable audio"})
                continue
            try:
                feats = extract_features(str(audio_path))
            except Exception as e:
                anomalies.append({"participant_id":pid,"emotion":emotion,"issue":f"feature extraction error: {e}"})
                continue
            valid_emotions += 1
            row = dict(participant_base)
            row.update({
                "emotion": emotion,
                "transcript": transcript,
                "audio_file": str(audio_path.relative_to(BASE)),
                "audio_source_type": source_type,
                "original_duration_sec": orig_dur,
                "original_sample_rate": orig_sr,
                "original_channels": orig_ch,
            })
            row.update(feats)
            rows.append(row)
            agg[f"{emotion}_transcript"] = transcript
            agg[f"{emotion}_audio_source_type"] = source_type
            for k,v in feats.items():
                agg[f"{emotion}_{k}"] = v
        if valid_emotions == 3:
            participant_rows.append(agg)
        else:
            anomalies.append({"participant_id":pid,"issue":f"only {valid_emotions}/3 usable emotion audios"})
    sample_df = pd.DataFrame(rows)
    part_df = pd.DataFrame(participant_rows)
    sample_path = OUT / f"EATD-{split}-features.csv"
    part_path = OUT / f"EATD-{split}-participant-features.csv"
    sample_df.to_csv(sample_path,index=False,encoding="utf-8-sig")
    part_df.to_csv(part_path,index=False,encoding="utf-8-sig")
    report = {
        "split": split,
        "source_folder": str(root),
        "participant_folders_total": len(pids),
        "usable_participants": int(len(part_df)),
        "sample_rows": int(len(sample_df)),
        "depressed_participants": int(part_df["depression_label"].sum()) if len(part_df) else 0,
        "non_depressed_participants": int((part_df["depression_label"]==0).sum()) if len(part_df) else 0,
        "cutoff_rule": "depression_label = 1 when standard SDS score >= 53",
        "audio_policy": "Prefer *_out.wav; fallback to raw *.wav only when preprocessed file is zero-duration/unusable",
        "feature_policy": "mono 16kHz load, peak amplitude normalization, 13 MFCC mean/std, pitch mean/std, RMS mean/std, ZCR mean/std, spectral centroid mean/std, duration",
        "anomalies": anomalies,
        "output_sample_csv": str(sample_path),
        "output_participant_csv": str(part_path),
    }
    (OUT / f"EATD-{split}-preprocessing-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines = [
        f"EATD {split.upper()} PREPROCESSING REPORT",
        "="*42,
        f"Participant folders total: {len(pids)}",
        f"Usable participants (all 3 emotion samples): {len(part_df)}",
        f"Audio-level feature rows: {len(sample_df)}",
        f"Depressed participants (SDS standard score >=53): {report['depressed_participants']}",
        f"Non-depressed participants: {report['non_depressed_participants']}",
        "",
        "Audio policy: prefer *_out.wav; fallback to raw *.wav only if *_out.wav is zero-duration/unusable.",
        "Text policy: preserve Chinese transcript wording; normalize whitespace only.",
        "Labels: raw label.txt preserved when available; new_label.txt used as standard SDS score.",
        "No labels were invented.",
        "",
        "Extracted features: 13 MFCC means + 13 MFCC stds, pitch mean/std, RMS mean/std,",
        "zero-crossing-rate mean/std, spectral centroid mean/std, duration.",
        "",
        "ANOMALIES:",
    ]
    if anomalies:
        lines += [json.dumps(a,ensure_ascii=False) for a in anomalies]
    else:
        lines.append("None")
    (OUT / f"EATD-{split}-preprocessing-report.txt").write_text("\n".join(lines),encoding="utf-8")
    return report

if __name__ == "__main__":
    reports = [process_split("train"), process_split("validation")]
    (OUT / "EATD-preprocessing-summary.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k!="anomalies"} | {"anomaly_count":len(r["anomalies"])} for r in reports],ensure_ascii=False,indent=2))
