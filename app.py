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
st.caption("Research screening prototype — NLP text signal + speech distress signal + documented late fusion")

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
            if result["speech"] is not None:
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
    "Current verified text model: TF-IDF + calibrated Linear SVM. "
    "Speech branch: EATD acoustic Logistic Regression selected using participant-grouped cross-validation. "
    "The combined decision layer is a research prototype and is not clinically calibrated."
)
