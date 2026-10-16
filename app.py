import re
import joblib
import streamlit as st

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Email / SMS Spam Detector")
st.write("Message paste karo aur check karo ki spam hai ya nahi.")

msg = st.text_area("Enter message")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Pehle kuch text daalo.")
    else:
        cleaned = re.sub(r"[^a-z\s]", "", msg.lower())
        result = model.predict(tfidf.transform([cleaned]))[0]
        if result == 1:
            st.error("Spam 🚫")
        else:
            st.success("Not Spam ✅")