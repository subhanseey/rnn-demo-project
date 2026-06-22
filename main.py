import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.keras', compile=False)

# Word index
word_index = imdb.get_word_index()

# Preprocess
def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)
        encoded_review.append(index + 3)
        
    padded_review = sequence.pad_sequences([encoded_review], maxlen=8)
    return padded_review

# Prediction
def predict_sentiment(text):
    data = preprocess_text(text)
    prediction = model.predict(data)

    score = float(prediction[0][0])
    sentiment = "Positive 😊" if score > 0.5 else "Negative 😐"

    return sentiment, score

# UI
st.title("IMDB Movie Review Sentiment Analysis")

user_input = st.text_area("Enter Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a review first")
    else:
        sentiment, score = predict_sentiment(user_input)
        st.write("Sentiment:", sentiment)
        st.write("Score:", score)
