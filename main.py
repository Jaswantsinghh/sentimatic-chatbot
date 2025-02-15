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

# Load chat history
CHAT_HISTORY_FILE = "chat_history.json"

# Predefined chatbot responses
responses = {
    1: ["That's wonderful! 😊", "I'm happy for you!", "Keep that positive energy going! 🚀"],
    0: ["I'm here for you. 🫂", "That sounds tough. Want to talk about it?", "You're not alone."],
    2: ["I see...", "Okay, tell me more.", "That's interesting!"]
}

# Function to simulate typing effect
def typing_effect(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()

# Function to classify sentiment using our trained model
def classify_sentiment(user_input):
    return sentiment_model.predict([user_input])[0]

# Function to translate text if needed
def translate_text(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

# Function to recall past conversations
def recall_past_conversation(user_input, history):
    for past_msg, response in history.items():
        if past_msg.lower() in user_input.lower():
            return f"I remember you said: '{past_msg}'. My response was: '{response}'"
    return None

# Function to generate responses
def generate_response(user_input, user_name, history):
    past_conversation = recall_past_conversation(user_input, history)
    if past_conversation:
        return past_conversation
    
    sentiment = classify_sentiment(user_input)
    return random.choice(responses[sentiment])

# Load chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}

# Save chat history
def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

# Chatbot Start
chat_history = load_chat_history()
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

    response = generate_response(translated_input, user_name, chat_history)
    chat_history[user_input] = response
    typing_effect(f"Chatbot: {response}")
