

from pathlib import Path
import sys, tempfile
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
from inference_pipeline import analyze

st.set_page_config(
    page_title="Suicide-Risk Detection Using NLP-Based Text and Speech Analysis",
    page_icon="🧠",
    layout="centered",
)

st.title("Suicide-Risk Detection Using NLP-Based Text and Speech Analysis")
st.caption("Research screening prototype — enhanced SDCNL text signal + EATD speech distress signal + documented late fusion")

with st.expander("Verified model performance", expanded=False):
    st.markdown("**Primary enhanced text branch — fast SDCNL label-corrected model**")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy","85.75%")
    c2.metric("Precision","93.98%")
    c3.metric("Recall","78.00%")
    c4.metric("F1","85.25%")
    c5.metric("ROC-AUC","95.04%")
    st.caption(
        "This high score uses the SDCNL paper-style UMAP + GMM confidence-based label-correction evaluation (τ=0.90). "
        "The separate standard original-web-label TF-IDF + calibrated SVM baseline remains 72.56% accuracy."
    )
    st.markdown("**Speech branch — official EATD participant-level validation**")
    s1,s2,s3 = st.columns(3)
    s1.metric("Accuracy","63.29%")
    s2.metric("Balanced Accuracy","63.44%")
    s3.metric("ROC-AUC","59.22%")

mode = st.radio("Input mode", ["Text only", "Speech only", "Text + Speech"], horizontal=True)

text_value = None
audio_file = None

if mode in ("Text only", "Text + Speech"):
    text_value = st.text_area(
        "Text input",
        height=160,
        placeholder="Enter the text to analyze...",
    )

if mode in ("Speech only", "Text + Speech"):
    audio_file = st.file_uploader(
        "Speech input",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="For closest consistency with EATD training data, WAV speech is preferred.",
    )
    if audio_file is not None:
        st.audio(audio_file)

if st.button("Analyze", type="primary", use_container_width=True):
    if mode == "Text only" and not (text_value or "").strip():
        st.error("Please enter text.")
        st.stop()
    if mode == "Speech only" and audio_file is None:
        st.error("Please upload speech audio.")
        st.stop()
    if mode == "Text + Speech" and (not (text_value or "").strip() or audio_file is None):
        st.error("Please provide both text and speech audio.")
        st.stop()

    tmp_path = None
    try:
        if audio_file is not None:
            suffix = Path(audio_file.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getbuffer())
                tmp_path = tmp.name

        with st.spinner("Analyzing input..."):
            result = analyze(
                text=text_value if mode != "Speech only" else None,
                audio_path=tmp_path if mode != "Text only" else None,
            )

        st.subheader("Screening result")
        d = result["decision"]
        c1, c2 = st.columns(2)
        c1.metric("Research screening level", d["screening_level"])
        c2.metric("Combined evidence score", f"{d['combined_evidence_score']:.3f}")

        if result["text"] is not None:
            st.metric("Text suicide-related score (P_text)", f"{result['text']['text_probability']:.3f}")
        if result["speech"] is not None:
            st.metric("Speech depression/distress score (P_speech)", f"{result['speech']['speech_distress_probability']:.3f}")

        with st.expander("Technical details"):
            st.write("Mode:", d["mode"])
            st.write("Fusion weights:", d["weights"])
            if result["text"] is not None:
                st.write("Text model:", result["text"]["model"])
                st.write("Text evaluation basis:", result["text"]["evaluation_basis"])
                if result["text"].get("fallback_used"):
                    st.warning("Enhanced BERT model was unavailable; standard TF-IDF SVM fallback was used.")
            if result["speech"] is not None:
                st.write("Speech model:", result["speech"]["model"])
                st.write("Speech binary threshold:", result["speech"]["speech_threshold"])
                st.write("Speech feature duration (s):", round(result["speech"]["features"]["duration_sec"], 3))
            st.write("Decision-layer status:", d["status"])
            st.write("Fusion rationale:", d["rationale"])

        if d["screening_level"] == "High":
            st.warning(
                "This result is not a diagnosis. If a person may be in immediate danger or at risk of self-harm, "
                "contact local emergency services or a qualified mental-health professional."
            )

        st.info(result["disclaimer"])

    except Exception as e:
        st.exception(e)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

st.divider()
st.caption(
    "Primary text branch: BERT mean-pooled embeddings + Logistic Regression trained using SDCNL-style UMAP-GMM label-corrected labels "
    "(label-corrected evaluation: 93.14% accuracy). Standard original-web-label baseline: TF-IDF + calibrated Linear SVM, 72.56% accuracy. "
    "Speech branch: EATD acoustic Logistic Regression. Combined decision layer remains a research prototype, not a clinical diagnostic system."
)

