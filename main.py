import joblib
import random
import time
import json
import os
import nltk
from deep_translator import GoogleTranslator
from collections import deque

nltk.download('punkt')

# Load trained sentiment model
sentiment_model = joblib.load("sentiment_model.pkl")

# Chat history file
CHAT_HISTORY_FILE = "chat_history.json"

# Maximum chat history to track progressive emotions
MAX_HISTORY = 5

# Enhanced chatbot responses with intensity levels
responses = {
    1: {  # Positive Sentiment
        "mild": ["That's nice to hear! 😊", "Glad to hear that!", "That's good!"],
        "moderate": ["That's wonderful! Keep it up! 🌟", "You're doing great!", "I love your energy! 🚀"],
        "strong": ["Wow! That sounds amazing! 🎉", "Incredible! So happy for you!", "Keep that excitement going! 🔥"]
    },
    0: {  # Negative Sentiment
        "mild": ["I understand. 🤗", "That doesn't sound great. You okay?", "That’s tough, but I'm here for you."],
        "moderate": ["I'm really sorry you're feeling that way. Want to talk more?", "You're not alone. I'm here for you. 🫂", "That sounds really difficult."],
        "strong": ["I’m really concerned about you. Please don’t hesitate to share. 💙", "I'm here to support you, no matter what. You're strong. 💪", "It sounds really hard. Want me to suggest something to help?"]
    },
    2: {  # Neutral Sentiment
        "mild": ["I see... Tell me more.", "Hmm, interesting.", "Got it!"],
        "moderate": ["That’s quite something! Keep going.", "I hear you. What’s on your mind?", "Sounds intriguing!"],
        "strong": ["Whoa! That’s deep. 🤯", "Now that’s something worth thinking about!", "That’s really thought-provoking!"]
    }
}

# Function to simulate typing effect
def typing_effect(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# Function to classify sentiment
def classify_sentiment(user_input):
    return sentiment_model.predict([user_input])[0]

# Function to determine emotional intensity
def get_intensity(sentiment_history):
    """Analyzes past emotions and determines intensity (mild, moderate, strong)."""
    if not sentiment_history:
        return "mild"
    
    # Count occurrences of the last sentiments
    recent_sentiments = list(sentiment_history)[-3:]  # Analyze last 3 messages
    positive_count = recent_sentiments.count(1)
    negative_count = recent_sentiments.count(0)

    # Determine intensity level based on trends
    if positive_count >= 3 or negative_count >= 3:
        return "strong"
    elif positive_count == 2 or negative_count == 2:
        return "moderate"
    else:
        return "mild"

# Function to recall past conversations
def recall_past_conversation(user_input, history):
    """Search for past messages and return a relevant response."""
    for past_msg, response, _ in history:
        if past_msg.lower() in user_input.lower():
            return f"I remember you said: '{past_msg}'. My response was: '{response}'"
    return None

# Function to translate text
def translate_text(text):
    """Translate text to English if needed"""
    try:
        translated_text = GoogleTranslator(source="auto", target="en").translate(text)
        return translated_text if translated_text else text  # Ensure non-empty output
    except Exception as e:
        print(f"Translation error: {e}")  # Debugging info
        return text  # Return original text if translation fails

# Function to generate chatbot responses
def generate_response(user_input, user_name, history, sentiment_history):
    """Generate a chatbot response based on sentiment, intensity, and past messages."""
    past_conversation = recall_past_conversation(user_input, history)
    if past_conversation:
        return past_conversation
    
    sentiment = classify_sentiment(user_input)
    sentiment_history.append(sentiment)  # Track sentiment progression

    # Determine response intensity based on past emotions
    intensity = get_intensity(sentiment_history)
    
    return random.choice(responses[sentiment][intensity])

# Load chat history (Ensuring tuples are stored)
def load_chat_history():
    """Load chat history from file and ensure data is correctly formatted."""
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            try:
                data = json.load(file)
                return deque([tuple(entry) for entry in data if isinstance(entry, list) and len(entry) == 3], maxlen=MAX_HISTORY)
            except json.JSONDecodeError:
                return deque(maxlen=MAX_HISTORY)  # Handle corrupt JSON file
    return deque(maxlen=MAX_HISTORY)

# Save chat history
def save_chat_history(history):
    """Save chat history to JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(list(history), file, indent=4)

# Chatbot Start
chat_history = load_chat_history()
sentiment_history = deque(maxlen=MAX_HISTORY)  # Store past emotions for progressive handling

typing_effect("Chatbot: Hello! What's your name?")
user_name = input("You: ").strip().title()

if not user_name:
    user_name = "Friend"

typing_effect(f"Chatbot: Nice to meet you, {user_name}! How are you feeling today?")

while True:
    user_input = input(f"{user_name}: ").strip()

    if user_input.lower() in ["exit", "bye", "quit"]:
        typing_effect("Chatbot: Are you sure you want to exit? (yes/no)")
        confirm_exit = input(f"{user_name}: ").strip().lower()
        if confirm_exit in ["yes", "y"]:
            typing_effect(f"Chatbot: Goodbye, {user_name}! Take care! 👋")
            save_chat_history(chat_history)
            break
        else:
            typing_effect("Chatbot: Great! Let's continue our chat.")
            continue

    if not user_input:
        typing_effect("Chatbot: Could you say that again?")
        continue

    translated_input = translate_text(user_input)
    if translated_input.lower() != user_input.lower():
        typing_effect(f"Chatbot: (Translated) {translated_input}")

    response = generate_response(translated_input, user_name, chat_history, sentiment_history)
    
    # Store only the last 5 messages (including sentiment)
    chat_history.append((user_input, response, classify_sentiment(translated_input)))

    typing_effect(f"Chatbot: {response}")
