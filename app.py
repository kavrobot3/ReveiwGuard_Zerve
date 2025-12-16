from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import re
import html
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="AI Review Detection API",
    description="Predict if a review is AI-generated and estimate regret score",
    version="1.0.0"
)

# Load the fixed models at startup
ai_detector = joblib.load('ai_detector_fixed.pkl')
regret_predictor = joblib.load('regret_predictor_fixed.pkl')

# Request model
class ReviewRequest(BaseModel):
    text: str = Field(..., description="Review text to analyze", min_length=1)
    rating: float = Field(default=3.0, description="Rating 1-5", ge=1.0, le=5.0)

# Response model
class ReviewResponse(BaseModel):
    is_ai_generated: bool
    ai_confidence_percent: float
    regret_score_1_10: float
    verdict: str

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
    """Extract all 10 features that match the trained models"""
    cleaned_text = clean_text(text)
    
    # Text statistics
    review_length = len(cleaned_text)
    words = cleaned_text.split()
    word_count = len(words)
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    unique_word_ratio = len(set(w.lower() for w in words)) / word_count if word_count > 0 else 0
    
    # VADER sentiment
    sentiment_polarity = calculate_vader_sentiment(cleaned_text)
    
    # Rating-sentiment mismatch
    # Expected sentiment based on rating (1=very negative, 5=very positive)
    expected_sentiment = (rating - 3) / 2  # Maps 1-5 to -1 to 1
    rating_sentiment_mismatch = abs(expected_sentiment - sentiment_polarity)
    
    # Punctuation features
    exclamation_count = cleaned_text.count('!')
    
    # Caps ratio
    letters = [c for c in cleaned_text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    
    # First-person pronouns
    text_lower = cleaned_text.lower()
    first_person_pronoun_count = (
        len(re.findall(r'\bi\b', text_lower)) +
        len(re.findall(r'\bmy\b', text_lower)) +
        len(re.findall(r'\bme\b', text_lower)) +
        len(re.findall(r'\bmine\b', text_lower))
    )
    pronoun_ratio = first_person_pronoun_count / word_count if word_count > 0 else 0.0
    
    # Return 10 features in exact order models expect
    return [
        review_length, word_count, avg_word_length, unique_word_ratio,
        sentiment_polarity, rating_sentiment_mismatch, exclamation_count,
        caps_ratio, first_person_pronoun_count, pronoun_ratio
    ]

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Review Detection API is running"}

@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
def predict(request: ReviewRequest):
    """Predict if a review is AI-generated and estimate regret score"""
    try:
        # Extract 10 features
        features = extract_features(request.text, request.rating)
        
        # Create DataFrame with correct feature names
        features_df = pd.DataFrame([features], columns=[
            'review_length', 'word_count', 'avg_word_length', 'unique_word_ratio',
            'sentiment_polarity', 'rating_sentiment_mismatch', 'exclamation_count',
            'caps_ratio', 'first_person_pronoun_count', 'pronoun_ratio'
        ])
        
        features_df = features_df.fillna(0)
        
        # Predictions
        is_ai = int(ai_detector.predict(features_df)[0])
        ai_proba = ai_detector.predict_proba(features_df)[0]
        ai_confidence = float(ai_proba[1] * 100)
        
        regret_score = float(regret_predictor.predict(features_df)[0])
        regret_score = max(1.0, min(10.0, regret_score))
        
        # Generate verdict
        if is_ai == 1:
            verdict = f"AI-generated ({ai_confidence:.1f}% confidence)"
        else:
            verdict = f"Human-written ({100-ai_confidence:.1f}% confidence)"
        
        return ReviewResponse(
            is_ai_generated=bool(is_ai),
            ai_confidence_percent=round(ai_confidence, 2),
            regret_score_1_10=round(regret_score, 2),
            verdict=verdict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
