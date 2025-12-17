import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import html
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Copy the helper functions from app.py
def clean_text(text_input):
    """Clean text by removing HTML tags, emojis, special chars"""
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

def calculate_vader_sentiment(text):
    """Calculate VADER sentiment score using weighted lexicon"""
    # VADER positive words with weights
    positive_words = {
        'great': 3.1, 'excellent': 3.2, 'amazing': 3.0, 'love': 3.2, 'perfect': 3.0,
        'best': 3.4, 'good': 2.9, 'awesome': 3.1, 'happy': 2.7, 'recommend': 2.5,
        'wonderful': 3.0, 'fantastic': 3.2, 'outstanding': 3.5, 'superb': 3.3,
        'delighted': 3.0, 'impressed': 2.8, 'satisfied': 2.5, 'pleased': 2.4,
        'enjoy': 2.1, 'like': 2.0, 'quality': 2.3, 'worth': 2.2, 'glad': 2.5
    }
    
    # VADER negative words with weights
    negative_words = {
        'bad': -2.5, 'worst': -3.1, 'terrible': -3.1, 'hate': -2.7, 'poor': -2.3,
        'awful': -3.0, 'disappointed': -2.5, 'waste': -2.4, 'broken': -2.2,
        'useless': -2.8, 'horrible': -2.9, 'pathetic': -2.6, 'terrible': -3.1,
        'fail': -2.2, 'defective': -2.4, 'return': -1.5, 'refund': -1.8,
        'terrible': -3.1, 'horrible': -2.9, 'junk': -2.3
    }
    
    text_lower = text.lower()
    words = text_lower.split()
    
    if not words:
        return 0.0
    
    compound_score = 0.0
    for word in words:
        if word in positive_words:
            compound_score += positive_words[word]
        elif word in negative_words:
            compound_score += negative_words[word]
    
    # Normalize to [-1, 1] range
    norm_score = compound_score / max(1, len(words))
    return max(-1.0, min(1.0, norm_score))

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

# Load your training data
# Assuming you have a CSV file with columns: category, rating, label, text_
# Replace 'fake reviews dataset.csv' with your actual file name
df = pd.read_csv('fake reviews dataset.csv')

# Process the data
df['is_ai'] = df['label'].apply(lambda x: 1 if x == 'CG' else 0)
df['regret_score'] = 11 - df['rating']  # Simple mapping: rating 1 -> 10, rating 5 -> 6
df = df.rename(columns={'text_': 'text'})

# Balance the dataset
min_count = min(df['is_ai'].value_counts())
df_balanced = pd.concat([
    df[df['is_ai'] == 0].sample(n=min_count, random_state=42),
    df[df['is_ai'] == 1].sample(n=min_count, random_state=42)
])

# Prepare cleaned texts for TF-IDF
cleaned_texts = df_balanced['text'].apply(clean_text)

# Fit TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
tfidf_features = tfidf_vectorizer.fit_transform(cleaned_texts)
tfidf_dense = tfidf_features.toarray()

# Extract statistical features
features_list = []
for _, row in df_balanced.iterrows():
    features = extract_features(row['text'], row['rating'])
    features_list.append(features)

statistical_features = pd.DataFrame(features_list, columns=[
    'length', 'word_count', 'simple_sentiment', 'sentiment_polarity', 'rating_sentiment_diff',
    'exclamation_count', 'question_count', 'caps_ratio', 'i_count', 'avg_word_length', 'unique_word_ratio'
])

# Combine features
combined_features = np.hstack([statistical_features.values, tfidf_dense])
features_df = pd.DataFrame(combined_features)

# Split data
X_train, X_test, y_ai_train, y_ai_test = train_test_split(features_df, df_balanced['is_ai'], test_size=0.2, random_state=42)
_, _, y_regret_train, y_regret_test = train_test_split(features_df, df_balanced['regret_score'], test_size=0.2, random_state=42)

# Train AI detector
ai_detector = XGBClassifier(n_estimators=500, random_state=42, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
ai_detector.fit(X_train, y_ai_train)

# Train regret predictor
regret_predictor = XGBRegressor(n_estimators=500, random_state=42, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
regret_predictor.fit(X_train, y_regret_train)

# Save models
joblib.dump(ai_detector, 'ai_detector_compatible.pkl')
joblib.dump(regret_predictor, 'regret_predictor_compatible.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("Models and vectorizer saved successfully!")

# Evaluate
ai_pred = ai_detector.predict(X_test)
regret_pred = regret_predictor.predict(X_test)

print(f"AI Detector Accuracy: {accuracy_score(y_ai_test, ai_pred)}")
print("AI Detector Classification Report:")
print(classification_report(y_ai_test, ai_pred, target_names=['Human', 'AI']))
print("Confusion Matrix:")
print(confusion_matrix(y_ai_test, ai_pred))

print(f"Regret Predictor MSE: {mean_squared_error(y_regret_test, regret_pred)}")
print(f"Regret Predictor MAE: {mean_absolute_error(y_regret_test, regret_pred)}")

# Save models
joblib.dump(ai_detector, 'ai_detector_compatible.pkl')
joblib.dump(regret_predictor, 'regret_predictor_compatible.pkl')

print("Models saved successfully!")