import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download model from Google Drive if not present
MODEL_PATH = 'model.h5'
FILE_ID = "14uRJ63AJ4AR4eo2RIPSGU3zYk5VeQlXJ"
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model, please wait...'):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

# Load model
model = load_model(MODEL_PATH, compile=False)

# Word index
word_index = imdb.get_word_index()

# Preprocess
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)
        encoded_review.append(index + 3)
    padded_review = pad_sequences([encoded_review], maxlen=500)
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
