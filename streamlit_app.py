import streamlit as st
import joblib
import pandas as pd
import re
import html

st.title("ReviewGuard – AI Review Detector & Regret Predictor")
st.write("Built by Kavish (13, India) — detects AI-generated reviews and predicts buyer regret")

# Load models
from huggingface_hub import hf_hub_download
import joblib

st.write("Loading models from Hugging Face... (first time takes a minute)")

ai_detector_path = hf_hub_download(repo_id="kavrobot/your-model-repo-name", filename="ai_detector_fixed.pkl")
regret_predictor_path = hf_hub_download(repo_id="kavrobot/your-model-repo-name", filename="regret_predictor_fixed.pkl")

ai_detector = joblib.load(ai_detector_path)
regret_predictor = joblib.load(regret_predictor_path)

st.success("Models loaded!")

# Clean text function
def clean_text(text_input):
    if not isinstance(text_input, str) or not text_input.strip():
        return text_input
    cleaned = html.unescape(text_input)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

# Extract features (copy your exact 9 features from app.py)
def extract_features(text, rating):
    cleaned_text = clean_text(text)
    length = len(cleaned_text)
    word_count = len(cleaned_text.split())
    positive_keywords = ['great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'good', 'awesome', 'happy', 'recommend']
    negative_keywords = ['bad', 'worst', 'terrible', 'hate', 'poor', 'awful', 'disappointed', 'waste', 'broken', 'useless']
    text_lower = cleaned_text.lower()
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    simple_sentiment = pos_count - neg_count
    rating_sentiment_diff = abs(rating - (simple_sentiment + 3))
    exclamation_count = cleaned_text.count('!')
    question_count = cleaned_text.count('?')
    letters = [c for c in cleaned_text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0
    i_count = len(re.findall(r'\bi\b', cleaned_text.lower()))
    return [rating, length, word_count, simple_sentiment, rating_sentiment_diff, exclamation_count, question_count, caps_ratio, i_count]

# UI
review_text = st.text_area("Paste a review here", height=150)
rating = st.slider("Rating (1-5)", 1.0, 5.0, 3.0)

if st.button("Analyze Review"):
    if review_text.strip():
        features = extract_features(review_text, rating)
        features_df = pd.DataFrame([features], columns=[
            'rating', 'length', 'word_count', 'simple_sentiment',
            'rating_sentiment_diff', 'exclamation_count', 'question_count',
            'caps_ratio', 'i_count'
        ])
        features_df = features_df.fillna(0)

        is_ai = int(ai_detector.predict(features_df)[0])
        ai_proba = ai_detector.predict_proba(features_df)[0]
        ai_confidence = float(ai_proba[1] * 100)

        regret_score = float(regret_predictor.predict(features_df)[0])
        regret_score = max(1.0, min(10.0, regret_score))

        st.success(f"**AI-Generated?** {'Yes' if is_ai else 'No'} ({ai_confidence:.1f}% confidence)")
        st.warning(f"**Regret Score (1-10):** {regret_score:.1f}")
        st.info(f"**Verdict:** {'High regret risk' if regret_score > 7 else 'Low regret risk'}")
    else:
        st.error("Please enter a review text")

st.caption("Live demo — try any Amazon-style review!")
