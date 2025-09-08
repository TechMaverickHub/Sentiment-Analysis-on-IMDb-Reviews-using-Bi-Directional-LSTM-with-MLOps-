import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import yaml

# Load config and model
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

model = tf.keras.models.load_model(config['model_path'])

# Load word index
from tensorflow.keras.datasets import imdb
word_index = imdb.get_word_index()

def encode_review(text, word_index, maxlen):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) for w in words]
    return pad_sequences([encoded], maxlen=maxlen)

st.title("Sentiment Analysis on Movie Reviews")
review = st.text_area("Enter a movie review:")

if st.button("Predict"):
    encoded = encode_review(review, word_index, config['maxlen'])
    prediction = model.predict(encoded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    st.write(f"**Sentiment:** {sentiment} ({prediction:.2f})")
