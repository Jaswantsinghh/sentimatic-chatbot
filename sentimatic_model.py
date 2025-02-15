import nltk
import pandas as pd
import random
import time
import json
import os
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (Use your own or download IMDB reviews dataset)
data = data = pd.read_csv("dataset/IMDB Dataset.csv")
# Keep only necessary columns
data = data[['review', 'sentiment']]
data = data.rename(columns={'review': 'message', 'sentiment': 'sentiment'})

# Convert labels to binary (Positive = 1, Negative = 0, Neutral = 2)
sentiment_map = {"positive": 1, "negative": 0, "neutral": 2}
data['sentiment'] = data['sentiment'].map(sentiment_map)

# Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['message'], data['sentiment'], test_size=0.2, random_state=42
)

# Define a pipeline (TF-IDF + Naïve Bayes)
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("classifier", MultinomialNB())
])

# Train the model
model_pipeline.fit(train_texts, train_labels)

# Evaluate
predictions = model_pipeline.predict(test_texts)
print("Model Accuracy:", accuracy_score(test_labels, predictions))

# Save the model
joblib.dump(model_pipeline, "sentiment_model.pkl")
