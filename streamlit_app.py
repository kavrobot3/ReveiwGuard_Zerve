import streamlit as st
import joblib
import pandas as pd
import re
import html

st.title("ReviewGuard – AI Review Detector & Regret Predictor")
st.write("Built by Kavish (13, India) — detects AI-generated reviews and predicts buyer regret")

# Load models
ai_detector = joblib.load('ai_detector_fixed.pkl')
regret_predictor = joblib.load('regret_predictor_fixed.pkl')

# Your clean_text and extract_features functions from app.py (copy them here)
def clean_text(text_input):
    if not isinstance(text_input, str) or not text_input.strip():
        return text_input
    cleaned = html.unescape(text_input)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" u"\U00002702-\U000027B0" u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def extract_features(text, rating):
    # Paste your exact extract_features code from app.py here (the 9 features)
    # ... (copy from your app.py)

# Input
review_text = st.text_area("Paste a review here", height=150)
rating = st.slider("Rating (1-5)", 1.0, 5.0, 3.0)

if st.button("Analyze Review"):
    if review_text.strip():
        features = extract_features(review_text, rating)
        features_df = pd.DataFrame([features], columns=[...])  # your column names

        is_ai = ai_detector.predict(features_df)[0]
        ai_conf = ai_detector.predict_proba(features_df)[0][1] * 100
        regret = regret_predictor.predict(features_df)[0]

        st.write(f"**AI-Generated?** {'Yes' if is_ai else 'No'} ({ai_conf:.1f}% confidence)")
        st.write(f"**Regret Score (1-10):** {regret:.1f}")
        st.write(f"**Verdict:** {'High regret risk' if regret > 7 else 'Low regret risk'}")
    else:
        st.write("Please enter a review")

st.write("Live demo — try any Amazon-style review!")
