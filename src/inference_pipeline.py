
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import librosa
from scipy.sparse import hstack

BASE = Path(__file__).resolve().parents[1]
TEXT_MODEL_DIR = BASE / "models" / "text_model"
ENHANCED_TEXT_DIR = BASE / "models" / "enhanced_text_model"
SPEECH_MODEL_DIR = BASE / "models" / "speech_model"
CONFIG_DIR = BASE / "config"

_TEXT_VEC = None
_TEXT_MODEL = None
_ENH_WORD_VEC = None
_ENH_CHAR_VEC = None
_ENH_TEXT_MODEL = None
_ENH_TEXT_CFG = None
_SPEECH_MODEL = None
_SPEECH_CFG = None
_FUSION_CFG = None

def _load_artifacts():
    global _TEXT_VEC, _TEXT_MODEL, _ENH_WORD_VEC, _ENH_CHAR_VEC, _ENH_TEXT_MODEL, _ENH_TEXT_CFG
    global _SPEECH_MODEL, _SPEECH_CFG, _FUSION_CFG

    if _TEXT_VEC is None:
        _TEXT_VEC = joblib.load(TEXT_MODEL_DIR / "tfidf_vectorizer.joblib")
    if _TEXT_MODEL is None:
        _TEXT_MODEL = joblib.load(TEXT_MODEL_DIR / "text_svm_calibrated.joblib")

    if (ENHANCED_TEXT_DIR / "calibrated_svm.joblib").exists():
        if _ENH_WORD_VEC is None:
            _ENH_WORD_VEC = joblib.load(ENHANCED_TEXT_DIR / "word_vectorizer.joblib")
        if _ENH_CHAR_VEC is None:
            _ENH_CHAR_VEC = joblib.load(ENHANCED_TEXT_DIR / "char_vectorizer.joblib")
        if _ENH_TEXT_MODEL is None:
            _ENH_TEXT_MODEL = joblib.load(ENHANCED_TEXT_DIR / "calibrated_svm.joblib")
        if _ENH_TEXT_CFG is None:
            _ENH_TEXT_CFG = json.loads((ENHANCED_TEXT_DIR / "config.json").read_text(encoding="utf-8"))

    if _SPEECH_MODEL is None:
        _SPEECH_MODEL = joblib.load(SPEECH_MODEL_DIR / "best_speech_single_audio_model.joblib")
    if _SPEECH_CFG is None:
        _SPEECH_CFG = json.loads((SPEECH_MODEL_DIR / "speech_single_audio_config.json").read_text(encoding="utf-8"))
    if _FUSION_CFG is None:
        _FUSION_CFG = json.loads((CONFIG_DIR / "fusion_config.json").read_text(encoding="utf-8"))

def predict_text(text: str, prefer_enhanced: bool = True):
    _load_artifacts()
    text = str(text or "").strip()
    if not text:
        raise ValueError("Text input is empty.")

    if prefer_enhanced and _ENH_TEXT_MODEL is not None:
        w = _ENH_WORD_VEC.transform([text])
        c = _ENH_CHAR_VEC.transform([text])
        cw = float(_ENH_TEXT_CFG["char_weight"])
        x = hstack([w, c * cw], format="csr")
        p = float(_ENH_TEXT_MODEL.predict_proba(x)[0, 1])
        return {
            "text_probability": p,
            "text_binary_prediction": int(p >= 0.5),
            "model": _ENH_TEXT_CFG["model"],
            "evaluation_basis": _ENH_TEXT_CFG["evaluation_basis"],
            "fallback_used": False,
        }

    x = _TEXT_VEC.transform([text])
    p = float(_TEXT_MODEL.predict_proba(x)[0, 1])
    return {
        "text_probability": p,
        "text_binary_prediction": int(p >= 0.5),
        "model": "TF-IDF bigram + calibrated Linear SVM",
        "evaluation_basis": "Original leakage-free SDCNL web labels",
        "fallback_used": True,
    }

def extract_audio_features(audio_path):
    audio_path = str(audio_path)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("Audio contains no samples.")
    mx = float(np.max(np.abs(y)))
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

def predict_speech(audio_path):
    _load_artifacts()
    feats = extract_audio_features(audio_path)
    cols = _SPEECH_CFG["all_input_features"]
    row = pd.DataFrame([[feats.get(c, np.nan) for c in cols]], columns=cols)
    p = float(_SPEECH_MODEL.predict_proba(row)[0, 1])
    thr = float(_SPEECH_CFG["threshold"])
    return {
        "speech_distress_probability": p,
        "speech_binary_prediction": int(p >= thr),
        "speech_threshold": thr,
        "model": _FUSION_CFG["speech_model"],
        "features": feats,
    }

def screening_band(score: float):
    _load_artifacts()
    low = float(_FUSION_CFG["low_upper_exclusive"])
    mod = float(_FUSION_CFG["moderate_upper_exclusive"])
    if score < low:
        return "Low"
    if score < mod:
        return "Moderate"
    return "High"

def combine_scores(text_probability=None, speech_probability=None):
    _load_artifacts()
    if text_probability is None and speech_probability is None:
        raise ValueError("At least one modality score is required.")

    if text_probability is not None and speech_probability is not None:
        tw = float(_FUSION_CFG["text_weight"])
        sw = float(_FUSION_CFG["speech_weight"])
        score = tw * float(text_probability) + sw * float(speech_probability)
        mode = "text+speech"
        weights = {"text": tw, "speech": sw}
    elif text_probability is not None:
        score = float(text_probability)
        mode = "text-only"
        weights = {"text": 1.0, "speech": 0.0}
    else:
        score = float(speech_probability)
        mode = "speech-only"
        weights = {"text": 0.0, "speech": 1.0}

    return {
        "mode": mode,
        "combined_evidence_score": score,
        "screening_level": screening_band(score),
        "weights": weights,
        "status": _FUSION_CFG["status"],
        "rationale": _FUSION_CFG["rationale"],
    }

def analyze(text=None, audio_path=None):
    t = predict_text(text) if text and str(text).strip() else None
    s = predict_speech(audio_path) if audio_path else None
    f = combine_scores(
        text_probability=t["text_probability"] if t else None,
        speech_probability=s["speech_distress_probability"] if s else None,
    )
    return {
        "text": t,
        "speech": s,
        "decision": f,
        "disclaimer": (
            "Research screening prototype only. The enhanced text branch is trained using SDCNL-style "
            "UMAP-GMM confidence-based label correction (tau=0.90). Its 85.75% result is a label-corrected "
            "evaluation and is reported separately from the original-web-label baseline. "
            "The speech branch uses EATD SDS-derived depression/distress audio rather than direct suicide labels. "
            "The modalities are not participant-matched, so the combined score is not a clinically validated "
            "suicide-risk probability or diagnosis."
        ),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Suicide-risk NLP + speech research screening inference")
    parser.add_argument("--text", default=None)
    parser.add_argument("--audio", default=None)
    args = parser.parse_args()
    print(json.dumps(analyze(args.text, args.audio), indent=2, ensure_ascii=False, default=float))
