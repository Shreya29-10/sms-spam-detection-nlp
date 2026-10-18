import json
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from signals import analyze_signals
import os
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from datetime import datetime

st.set_page_config(page_title="SpamShield | NLP Spam Detector",
                   page_icon="🛡️", layout="centered")

MODEL_FILES = {
    "Naive Bayes": "model_nb.pkl",
    "Logistic Regression": "model_lr.pkl",
    "Linear SVM": "model_svm.pkl",
}

@st.cache_resource
def load_all():
    models = {name: joblib.load(f) for name, f in MODEL_FILES.items()}
    return models, joblib.load("tfidf.pkl")

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

models, tfidf = load_all()
metrics = load_metrics()

def clean(text):
    return re.sub(r"[^a-z\s]", "", str(text).lower())

def spam_probabilities(model, texts):
    vec = tfidf.transform([clean(t) for t in texts])
    return model.predict_proba(vec)[:, 1]

def spam_words(model, vec, top=6):
    """Message ke wo words jo model ko spam ki taraf dhakel rahe hain."""
    try:
        names = tfidf.get_feature_names_out()
        idx = vec.nonzero()[1]
        if len(idx) == 0:
            return []
        if hasattr(model, "feature_log_prob_"):
            w = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        elif hasattr(model, "coef_"):
            w = model.coef_[0]
        else:
            w = np.mean([c.estimator.coef_[0]
                         for c in model.calibrated_classifiers_], axis=0)
        pairs = sorted(zip(names[idx], w[idx]), key=lambda x: -x[1])
        return [word for word, val in pairs if val > 0][:top]
    except Exception:
        return []
@st.cache_data
def load_dataset():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": "Not Spam", "spam": "Spam"})
    df["chars"] = df["text"].str.len()
    df["words"] = df["text"].str.split().str.len()
    return df

@st.cache_data
def top_words(label, n=10):
    df = load_dataset()
    texts = df[df["label"] == label]["text"].apply(clean)
    stop = list(ENGLISH_STOP_WORDS.union({"gt", "lt", "amp", "ll", "ve"}))
    cv = CountVectorizer(stop_words=stop)
    counts = cv.fit_transform(texts).sum(axis=0).A1
    words = cv.get_feature_names_out()
    return pd.Series(counts, index=words).sort_values(ascending=False).head(n)

def show_words(series, container):
    d = series.reset_index()
    d.columns = ["Word", "Count"]
    container.dataframe(
        d, hide_index=True, use_container_width=True,
        column_config={"Count": st.column_config.ProgressColumn(
            "Count", min_value=0, max_value=int(d["Count"].max()), format="%d")})
FEEDBACK_FILE = "feedback.csv"

def give_feedback(is_correct):
    last = st.session_state["last"]
    predicted = "Spam" if last["prob"] >= 0.5 else "Not Spam"
    opposite = "Not Spam" if predicted == "Spam" else "Spam"
    row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": last["msg"],
        "model": last["model"],
        "predicted": predicted,
        "spam_probability_%": round(last["prob"] * 100, 1),
        "user_says_correct": is_correct,
        "true_label": predicted if is_correct else opposite,
    }])
    row.to_csv(FEEDBACK_FILE, mode="a",
               header=not os.path.exists(FEEDBACK_FILE), index=False)
    st.session_state["last"]["feedback"] = is_correct
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
    st.header("⚙️ Settings")
    choice = st.selectbox("Choose model", list(MODEL_FILES.keys()))
    model = models[choice]
    m = metrics[choice]

    st.markdown("**Selected model stats**")
    st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    c1, c2 = st.columns(2)
    c1.metric("Precision", f"{m['precision']*100:.1f}%")
    c2.metric("Recall", f"{m['recall']*100:.1f}%")
    st.metric("F1-score", f"{m['f1']*100:.1f}%")

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown("1. Text cleaning\n2. TF-IDF features\n3. Classifier")
    st.caption("Dataset: 5,572 messages")
    st.caption("TE NLP Mini Project | AI & DS Engineering")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🛡️ SpamShield</h1>
  <p>Paste any SMS or email text and find out instantly if it is spam.</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_batch, tab_compare, tab_insights = st.tabs(
    ["🔍 Detect", "📂 Batch Analysis", "📊 Model Comparison", "💡 Insights"])

# =====================  TAB 1: DETECT  =====================
with tab_detect:
    SPAM_EX = "Congratulations! You have won a free prize. Call now to claim your 1000 cash reward."
    HAM_EX = "Hey, are we meeting at college tomorrow for the project work?"

    if "msg" not in st.session_state:
        st.session_state.msg = ""

    def set_msg(text):
        st.session_state.msg = text

    b1, b2 = st.columns(2)
    b1.button("🚨 Try a spam example", on_click=set_msg, args=(SPAM_EX,), use_container_width=True)
    b2.button("✅ Try a normal example", on_click=set_msg, args=(HAM_EX,), use_container_width=True)

    msg = st.text_area("Your message", key="msg", height=160,
                       placeholder="Type or paste the message here...")

    if st.button("🔍 Analyze Message", type="primary", use_container_width=True):
        if msg.strip() == "":
            st.session_state.pop("last", None)
            st.warning("Please enter some text first.")
        else:
            cleaned = clean(msg)
            prob = float(model.predict_proba(tfidf.transform([cleaned]))[0][1])
            st.session_state["last"] = {"msg": msg, "cleaned": cleaned, "prob": prob,
                                        "model": choice, "feedback": None}

    last = st.session_state.get("last")
    if last:
        used_model = models[last["model"]]
        vec = tfidf.transform([last["cleaned"]])
        spam_prob = last["prob"]
        is_spam = spam_prob >= 0.5
        confidence = spam_prob if is_spam else 1 - spam_prob

        preview = last["msg"][:70] + ("..." if len(last["msg"]) > 70 else "")
        st.caption(f"Result for: {preview}")

        if is_spam:
            st.markdown(f"""<div class="result spam">
                <h3>🚫 Spam Detected</h3>
                <span>Model: {last['model']} | Confidence: {confidence*100:.1f}%</span></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result ham">
                <h3>✅ Looks Safe (Not Spam)</h3>
                <span>Model: {last['model']} | Confidence: {confidence*100:.1f}%</span></div>""",
                        unsafe_allow_html=True)

        st.write("")
        st.caption("Spam probability")
        st.progress(spam_prob)

        sig_list = analyze_signals(last["msg"])
        hits = [s for s in sig_list if s["found"]]
        st.markdown(f"**🚩 Suspicious signals: {len(hits)} of {len(sig_list)} detected**")
        sc = st.columns(2)
        for i, s in enumerate(sig_list):
            icon = "🔴" if s["found"] else "🟢"
            extra = f" - `{s['detail']}`" if s["found"] and s["detail"] else ""
            sc[i % 2].markdown(f"{icon} **{s['name']}**{extra}")
        st.caption("Signals are rule-based hints, separate from the ML model's prediction.")

        words = spam_words(used_model, vec)
        if words:
            st.markdown("**Words that look like spam:**")
            st.markdown("".join(f'<span class="chip">{w}</span>' for w in words),
                        unsafe_allow_html=True)

        with st.expander("🆚 Compare all models on this message"):
            rows = []
            for name, mdl in models.items():
                p = float(mdl.predict_proba(vec)[0][1])
                rows.append({"Model": name,
                             "Spam probability %": round(p * 100, 1),
                             "Prediction": "Spam" if p >= 0.5 else "Not Spam"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("See processed text"):
            st.code(last["cleaned"])

        # ---------- Feedback ----------
        st.divider()
        if last["feedback"] is None:
            st.markdown("**Was this prediction correct?**")
            f1, f2 = st.columns(2)
            f1.button("👍 Yes, correct", on_click=give_feedback, args=(True,),
                      use_container_width=True)
            f2.button("👎 No, it's wrong", on_click=give_feedback, args=(False,),
                      use_container_width=True)
        else:
            st.success("Thanks! Your feedback was saved.")

    with st.expander("📝 Feedback log"):
        if os.path.exists(FEEDBACK_FILE):
            fb = pd.read_csv(FEEDBACK_FILE)
            wrong = int((fb["user_says_correct"] == False).sum())
            q1, q2 = st.columns(2)
            q1.metric("Feedback received", len(fb))
            q2.metric("Marked wrong", wrong)
            st.dataframe(fb, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download feedback CSV",
                               fb.to_csv(index=False).encode("utf-8"),
                               file_name="feedback.csv", mime="text/csv")
        else:
            st.caption("No feedback yet. Analyze a message and use the buttons below the result.")

# =====================  TAB 2: BATCH  =====================
with tab_batch:
    st.subheader("Batch Analysis")
    st.write(f"Upload a CSV file with many messages. Model in use: **{choice}**")

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
            probs = spam_probabilities(model, data[text_col].fillna("").tolist())
            result = data.copy()
            result["spam_probability_%"] = (probs * 100).round(1)
            result["prediction"] = ["Spam" if p >= 0.5 else "Not Spam" for p in probs]
            result["suspicious_signals"] = [sum(s["found"] for s in analyze_signals(t))
                                            for t in data[text_col].fillna("").tolist()]
            st.session_state["batch_result"] = result

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        total = len(result)
        n_spam = int((result["prediction"] == "Spam").sum())

        r1, r2, r3 = st.columns(3)
        r1.metric("Total messages", total)
        r2.metric("Spam", n_spam)
        r3.metric("Not Spam", total - n_spam)

        st.bar_chart(result["prediction"].value_counts())
        st.dataframe(result, use_container_width=True)

        st.download_button("⬇️ Download results as CSV",
                           result.to_csv(index=False).encode("utf-8"),
                           file_name="spam_analysis_results.csv",
                           mime="text/csv", use_container_width=True)

# =====================  TAB 3: MODEL COMPARISON  =====================
with tab_compare:
    st.subheader("Model Comparison")
    st.write("All three models were trained on the same data and tested on the same 20% held-out messages.")

    comp = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1"]] * 100
    comp.columns = ["Accuracy %", "Precision %", "Recall %", "F1-score %"]
    comp = comp.round(2)

    st.dataframe(comp, use_container_width=True)
    st.bar_chart(comp)

    best = comp["F1-score %"].idxmax()
    st.success(f"Best F1-score: **{best}** ({comp.loc[best, 'F1-score %']}%)")
    st.caption("Precision = of all messages flagged spam, how many really were spam. "
               "Recall = of all real spam, how many were caught. "
               "F1 balances both, which matters more than accuracy on imbalanced data.")

    st.markdown(f"**Confusion matrix counts: {choice}**")
    mm = metrics[choice]
    cm = pd.DataFrame([[mm["tn"], mm["fp"]], [mm["fn"], mm["tp"]]],
                      index=["Actual Not Spam", "Actual Spam"],
                      columns=["Predicted Not Spam", "Predicted Spam"])
    st.dataframe(cm, use_container_width=True)
# =====================  TAB 4: INSIGHTS  =====================
with tab_insights:
    st.subheader("Dataset & Model Insights")
    df_all = load_dataset()

    # ---- 1. Class distribution ----
    st.markdown("#### 1. Class distribution")
    counts = df_all["label"].value_counts()
    total = len(df_all)
    spam_pct = counts["Spam"] / total * 100
    k1, k2, k3 = st.columns(3)
    k1.metric("Total messages", f"{total:,}")
    k2.metric("Not Spam", f"{counts['Not Spam']:,}")
    k3.metric("Spam", f"{counts['Spam']:,}")
    st.bar_chart(counts)
    st.caption(f"Only {spam_pct:.1f}% of the messages are spam, so the data is imbalanced. "
               "That is why precision, recall and F1 matter more than accuracy alone.")

    # ---- 2. Message length ----
    st.markdown("#### 2. Message length: spam vs normal")
    avg = df_all.groupby("label")[["chars", "words"]].mean().round(1)
    avg.columns = ["Avg characters", "Avg words"]
    st.dataframe(avg, use_container_width=True)

    bins = list(range(0, 320, 20))
    hist = pd.DataFrame(index=bins[:-1])
    for lab in ["Not Spam", "Spam"]:
        vals = df_all[df_all["label"] == lab]["chars"].clip(upper=299)
        c, _ = np.histogram(vals, bins=bins)
        hist[lab] = (c / c.sum() * 100).round(1)
    hist.index.name = "Message length (characters)"
    st.bar_chart(hist)
    st.caption(f"Spam messages average {avg.loc['Spam', 'Avg characters']:.0f} characters, "
               f"normal messages {avg.loc['Not Spam', 'Avg characters']:.0f}. "
               "Y-axis shows the % of messages in each class; the last bucket includes longer messages.")

    # ---- 3. Top words ----
    st.markdown("#### 3. Most frequent words")
    w1, w2 = st.columns(2)
    w1.markdown("**🚫 Top spam words**")
    show_words(top_words("Spam"), w1)
    w2.markdown("**✅ Top normal-message words**")
    show_words(top_words("Not Spam"), w2)

    # ---- 4. Model performance ----
    st.markdown("#### 4. Model performance")
    left, right = st.columns(2)
    if os.path.exists("confusion_matrix.png"):
        left.image("confusion_matrix.png",
                   caption="Confusion matrix - Naive Bayes (20% test set)")
    right.markdown("**F1-score by model (%)**")
    f1s = pd.Series({n: v["f1"] * 100 for n, v in metrics.items()}).round(2)
    right.bar_chart(f1s)

    with st.expander("See raw data sample"):
        st.dataframe(df_all[["label", "text"]].sample(10, random_state=1),
                     use_container_width=True, hide_index=True)