import nltk
import pandas as pd
import re
import joblib
import os
from collections import deque
from bs4 import BeautifulSoup  # To clean HTML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset from multiple sources
datasets = [
    "dataset/IMDB Dataset.csv",       # Movie reviews
    "dataset/Twitter_Data.csv",       # Social media tweets
    "dataset/Reddit_Data.csv"         # Forum discussions
]

# Read and merge datasets
dataframes = []
for file in datasets:
    if os.path.exists(file):
        df = pd.read_csv(file)
        print(f"Loaded dataset: {file}")
        print("Columns:", df.columns)
        dataframes.append(df)

# Merge all datasets into one
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
else:
    print("No valid datasets found.")
    exit()

# Rename IMDB's 'review' column to 'message' to maintain consistency
if "review" in data.columns:
    data = data.rename(columns={"review": "message"})

# Identify Twitter/Reddit dataset and rename columns
if "clean_text" in data.columns and "category" in data.columns:
    print("Twitter/Reddit dataset detected.")
    data = data.rename(columns={"clean_text": "message", "category": "sentiment"})

# Fix: Check for duplicate sentiment columns and remove extra ones
if list(data.columns).count("sentiment") > 1:
    data = data.loc[:, ~data.columns.duplicated()]  # Remove duplicate column
    print("Fixed duplicate sentiment column.")

# Debug: Check final column names before mapping
print("Columns in merged dataset after fix:", data.columns)

# Standardize sentiment labels:
sentiment_map = {
    "positive": 1, "negative": 0, "neutral": 2,  # IMDB, Amazon, Reddit
    "-1": 0, "0": 2, "1": 1,  # Twitter (-1, 0, 1) → (0, 2, 1)
    -1: 0, 0: 2, 1: 1  # Ensure numerical mapping also works
}

# Ensure sentiment column exists before mapping
if "sentiment" in data.columns:
    data['sentiment'] = data['sentiment'].astype(str).str.lower().map(sentiment_map)
else:
    print("Error: 'sentiment' column not found in the dataset. Check dataset format.")
    exit()

# Drop missing or unmapped sentiment values
data = data.dropna(subset=['sentiment'])

# Convert back to integer after mapping
data['sentiment'] = data['sentiment'].astype(int)

# Text Preprocessing
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return None  # Return None if text is missing
    
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML (IMDB reviews)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)  # Convert hashtags to words (e.g., #modi → modi)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\b\w{1,2}\b", "", text)  # Remove very short words (1-2 letters)
    text = text.strip()
    
    return text if len(text.split()) > 3 else None  # Ensure at least 3 meaningful words remain

# Apply text preprocessing
data["message"] = data["message"].apply(preprocess_text)

# Drop empty messages after preprocessing
data = data.dropna(subset=["message"])

# Debug: Show 10 preprocessed samples
print("\nSAMPLE TRAINING DATA AFTER PREPROCESSING:")
print(data["message"].head(10))

# **If data is still empty, exit**
if data.shape[0] == 0:
    print("\nError: No valid text data found after preprocessing!")
    exit()

# Train/Test Split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['message'], data['sentiment'], test_size=0.2, random_state=42, stratify=data['sentiment']
)

# Define a pipeline (TF-IDF + Logistic Regression)
model_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=8000, ngram_range=(1,2))),
    ("classifier", LogisticRegression(max_iter=200))
])

# Train the model
print("Training the model...")
model_pipeline.fit(train_texts, train_labels)

# Evaluate the model
predictions = model_pipeline.predict(test_texts)
accuracy = accuracy_score(test_labels, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Save the model
joblib.dump(model_pipeline, "sentiment_model.pkl")
print("Model saved as 'sentiment_model.pkl'")
