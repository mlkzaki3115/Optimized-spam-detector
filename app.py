from pathlib import Path
import re
import string
import joblib
import streamlit as st


st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.block-container {
    max-width: 850px;
    padding-top: 3rem;
}
.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    text-align: center;
    margin-bottom: 2rem;
}
.card {
    padding: 1.5rem;
    border-radius: 16px;
    background: #1e293b;
    border: 1px solid #334155;
    margin-top: 1rem;
}
h1, h2, h3, p, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🛡️ SpamShield</h1>
    <p>Intelligent SMS spam detection powered by a tuned SVM model</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python main.py` before starting the app.")
    st.stop()

try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["model"]
    vectorizer = artifacts["vectorizer"]
    accuracy = artifacts.get("accuracy")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


def clean_message(message):
    message = message.lower()
    message = re.sub(
        f"[{re.escape(string.punctuation)}]", "", message
    )
    message = re.sub(r"\d+", "", message)
    return re.sub(r"\s+", " ", message).strip()


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📩 Analyze a message")

message = st.text_area(
    "Paste or type an SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize...",
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        features = vectorizer.transform([clean_message(message)])
        prediction = str(model.predict(features)[0]).lower()

        if hasattr(model, "decision_function"):
            score = float(model.decision_function(features)[0])
            confidence = min(abs(score) / 3, 1)
        else:
            confidence = None

        if prediction == "spam":
            st.error("🚨 This message is likely SPAM.")
        else:
            st.success("✅ This message appears to be HAM.")

        if confidence is not None:
            st.progress(confidence, text=f"Model confidence: {confidence:.1%}")

st.markdown("</div>", unsafe_allow_html=True)

if accuracy is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("📊 Test Accuracy", f"{accuracy:.2%}")
    st.caption("Accuracy measured on the held-out test dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("SpamShield • Tuned LinearSVC text classifier")