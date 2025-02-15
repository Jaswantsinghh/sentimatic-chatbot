# 🧠 Sentimatic Chatbot - Custom Sentiment Analysis Chatbot

## 🚀 Overview
This is a **sentiment-aware chatbot** that remembers past conversations and analyzes user sentiment **without using VADER**. Instead, it uses a **custom Naïve Bayes sentiment model**, trained on real-world datasets.

## 🛠 Features
✅ **Custom Sentiment Model** (Trained using Naïve Bayes + TF-IDF)  
✅ **Remembers Past Conversations** (Stored in `chat_history.json`)  
✅ **Multilingual Support** (Auto-translates messages to English)  
✅ **Personalized Responses** (Uses historical data for better replies)  
✅ **Fast & Lightweight** (No heavy deep learning dependencies)  

---

## 📥 Installation Guide

### **1️⃣ Install Dependencies**
Make sure you have Python 3.8+ installed. Then run:

```bash
git clone https://github.com/jaswantsinghh/sentimatic-chatbot.git
cd sentimatic-chatbot
python main.py

