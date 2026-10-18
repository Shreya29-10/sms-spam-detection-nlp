import re
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SpamShield | NLP Spam Detector",
                   page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("tfidf.pkl")

model, tfidf = load_model()

def clean(text):
    return re.sub(r"[^a-z\s]", "", str(text).lower())

def spam_probabilities(texts):
    vec = tfidf.transform([clean(t) for t in texts])
    return model.predict_proba(vec)[:, 1]

st.markdown("""
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed 60%,#db2777);
       padding: 2rem; border-radius: 18px; color: white; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2.2rem; color: white;}
.hero p {margin: .4rem 0 0; opacity: .9;}
.result {padding: 1.2rem 1.4rem; border-radius: 14px; margin-top: 1rem; border: 1px solid;}
.result h3 {margin: 0 0 .2rem 0;}
.spam {background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.55);}
.ham  {background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.55);}
.chip {display: inline-block; padding: .2rem .7rem; margin: .15rem;
       border-radius: 999px; background: rgba(124,58,237,.18);
       border: 1px solid rgba(124,58,237,.5); font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About Project")
    st.write("Email / SMS spam detection using NLP and Machine Learning.")
    st.markdown("**Pipeline**")
    st.markdown("1. Text cleaning\n2. TF-IDF features\n3. Naive Bayes classifier")
    st.markdown("**Model Stats**")
    st.metric("Accuracy", "96.23%")
    st.metric("Dataset size", "5,572 messages")
    st.caption("TE NLP Mini Project | AI & DS Engineering")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🛡️ SpamShield</h1>
  <p>Paste any SMS or email text and find out instantly if it is spam.</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_batch = st.tabs(["🔍 Detect", "📂 Batch Analysis"])

# =====================  TAB 1: DETECT  =====================
with tab_detect:
    SPAM_EX = "Congratulations! You have won a free prize. Call now to claim your 1000 cash reward."
    HAM_EX = "Hey, are we meeting at college tomorrow for the project work?"

    if "msg" not in st.session_state:
        st.session_state.msg = ""

    def set_msg(text):
        st.session_state.msg = text

    c1, c2 = st.columns(2)
    c1.button("🚨 Try a spam example", on_click=set_msg, args=(SPAM_EX,), use_container_width=True)
    c2.button("✅ Try a normal example", on_click=set_msg, args=(HAM_EX,), use_container_width=True)

    msg = st.text_area("Your message", key="msg", height=160,
                       placeholder="Type or paste the message here...")

    if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
        if msg.strip() == "":
            st.warning("Please enter some text first.")
        else:
            cleaned = clean(msg)
            vec = tfidf.transform([cleaned])
            spam_prob = float(model.predict_proba(vec)[0][1])
            is_spam = spam_prob >= 0.5
            confidence = spam_prob if is_spam else 1 - spam_prob

            if is_spam:
                st.markdown(f"""<div class="result spam">
                    <h3>🚫 Spam Detected</h3>
                    <span>Confidence: {confidence*100:.1f}%</span></div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result ham">
                    <h3>✅ Looks Safe (Not Spam)</h3>
                    <span>Confidence: {confidence*100:.1f}%</span></div>""",
                            unsafe_allow_html=True)

            st.write("")
            st.caption("Spam probability")
            st.progress(spam_prob)

            names = tfidf.get_feature_names_out()
            idx = vec.nonzero()[1]
            if len(idx) > 0:
                diff = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
                spammy = sorted(zip(names[idx], diff), key=lambda x: -x[1])
                spammy = [w for w, d in spammy if d > 0][:6]
                if spammy:
                    st.markdown("**Words that look like spam:**")
                    st.markdown("".join(f'<span class="chip">{w}</span>' for w in spammy),
                                unsafe_allow_html=True)

            with st.expander("See processed text"):
                st.code(cleaned)

# =====================  TAB 2: BATCH  =====================
with tab_batch:
    st.subheader("Batch Analysis")
    st.write("Upload a CSV file with many messages and check all of them at once.")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file is not None:
        try:
            data = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            data = pd.read_csv(file, encoding="latin-1")

        st.caption(f"{len(data)} rows loaded. Preview:")
        st.dataframe(data.head(), use_container_width=True)

        text_col = st.selectbox("Which column contains the messages?", data.columns)

        if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
            probs = spam_probabilities(data[text_col].fillna("").tolist())
            result = data.copy()
            result["spam_probability_%"] = (probs * 100).round(1)
            result["prediction"] = ["Spam" if p >= 0.5 else "Not Spam" for p in probs]
            st.session_state["batch_result"] = result

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        total = len(result)
        n_spam = int((result["prediction"] == "Spam").sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("Total messages", total)
        m2.metric("Spam", n_spam)
        m3.metric("Not Spam", total - n_spam)

        st.bar_chart(result["prediction"].value_counts())
        st.dataframe(result, use_container_width=True)

        st.download_button("⬇️ Download results as CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           file_name="spam_analysis_results.csv",
                           mime="text/csv", use_container_width=True)