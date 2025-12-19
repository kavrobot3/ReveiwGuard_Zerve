import streamlit as st
import joblib
import pandas as pd
import re
import html
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("ReviewGuard – AI Review Detector & Regret Predictor")

# Display logo in center below title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("FullLogo_Transparent.png", width=200)

st.write("Built by Kavish (Grade 8) — detects AI-generated reviews and predicts buyer regret")

# Load the compatible models directly
ai_detector = joblib.load('ai_detector_compatible.pkl')
regret_predictor = joblib.load('regret_predictor_compatible.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

def clean_text(text_input):
    if not isinstance(text_input, str) or not text_input.strip():
        return ""
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
    """Extract all 11 features that match the trained models"""
    cleaned_text = clean_text(text)
    
    # Text statistics
    review_length = len(cleaned_text)
    words = cleaned_text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    unique_word_ratio = len(set(w.lower() for w in words)) / word_count if word_count > 0 else 0
    
    # Simple sentiment: weighted sum of positive minus negative keywords
    positive_weights = {
        'great': 3.1, 'excellent': 3.2, 'amazing': 3.0, 'love': 3.2, 'perfect': 3.0,
        'best': 3.4, 'good': 2.9, 'awesome': 3.1, 'happy': 2.7, 'recommend': 2.5
    }
    negative_weights = {
        'bad': -2.5, 'worst': -3.1, 'terrible': -3.1, 'hate': -2.7, 'poor': -2.3,
        'awful': -3.0, 'disappointed': -2.5, 'waste': -2.4, 'broken': -2.2,
        'useless': -2.8, 'horrible': -2.9, 'pathetic': -2.6, 'terrible': -3.1,
        'fail': -2.2, 'defective': -2.4, 'return': -1.5, 'refund': -1.8,
        'terrible': -3.1, 'horrible': -2.9, 'junk': -2.3
    }
    simple_sentiment = 0
    for word in words:
        if word in positive_weights:
            simple_sentiment += positive_weights[word]
        elif word in negative_weights:
            simple_sentiment += negative_weights[word]
    
    # VADER sentiment
    vader_scores = vader_analyzer.polarity_scores(cleaned_text)
    sentiment_polarity = vader_scores['compound']
    
    # Rating-sentiment mismatch
    # Expected sentiment based on rating (1=very negative, 5=very positive)
    expected_sentiment = (rating - 3) / 2  # Maps 1-5 to -1 to 1
    rating_sentiment_diff = abs(expected_sentiment - sentiment_polarity)
    
    # Punctuation features
    exclamation_count = cleaned_text.count('!')
    question_count = cleaned_text.count('?')
    
    # Caps ratio
    letters = [c for c in cleaned_text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    
    # I count
    text_lower = cleaned_text.lower()
    i_count = len(re.findall(r'\bi\b', text_lower))
    
    # Return 11 features
    return [
        review_length, word_count, simple_sentiment, sentiment_polarity, rating_sentiment_diff,
        exclamation_count, question_count, caps_ratio, i_count, avg_word_length, unique_word_ratio
    ]

# UI
review_text = st.text_area("Paste a review here", height=150)
rating = st.slider("Rating (1-5)", 1.0, 5.0, 3.0)

if st.button("Analyze Review"):
    if review_text.strip():
        features = extract_features(review_text, rating)
        cleaned_text = clean_text(review_text)
        tfidf_features = tfidf_vectorizer.transform([cleaned_text]).toarray()
        combined_features = np.concatenate([features, tfidf_features.flatten()])
        features_df = pd.DataFrame([combined_features], columns=[str(i) for i in range(len(combined_features))])
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
