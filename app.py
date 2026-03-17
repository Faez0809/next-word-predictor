import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
# -------------------------
# Load model & files
# -------------------------
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

# -------------------------
# Theme Toggle
# -------------------------
dark_mode = st.toggle("Dark Mode", value=True)

if dark_mode:
    bg = "#0b1220"
    card = "#111827"
    text = "#e5e7eb"
    subtext = "#9ca3af"
    accent = "#3b82f6"
else:
    bg = "#f9fafb"
    card = "#ffffff"
    text = "#111827"
    subtext = "#6b7280"
    accent = "#2563eb"

# -------------------------
# Global Styling (FULL PAGE FIX)
# -------------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    background-color: {bg} !important;
    color: {text} !important;
}}

.block-container {{
    padding-top: 3rem;
}}

.container {{
    background-color: {card};
    padding: 40px;
    border-radius: 12px;
    max-width: 700px;
    margin: auto;
}}

.title {{
    font-size: 34px;
    font-weight: 600;
    margin-bottom: 5px;
}}

.subtitle {{
    font-size: 14px;
    color: {subtext};
    margin-bottom: 25px;
}}

.prediction {{
    font-size: 22px;
    font-weight: 500;
    color: {accent};
    margin-top: 20px;
}}

.stTextInput > div > div > input {{
    font-size: 16px;
    padding: 12px;
    border-radius: 8px;
}}

.stButton button {{
    width: 100%;
    border-radius: 8px;
    padding: 10px;
    font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Prediction function
# -------------------------
def predict_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding='pre')

    pred = model.predict(seq, verbose=0)
    index = np.argmax(pred)

    for word, i in tokenizer.word_index.items():
        if i == index:
            return word
    return ""

# -------------------------
# UI Layout
# -------------------------
st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown('<div class="title">Next Word Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LSTM-based text completion</div>', unsafe_allow_html=True)

user_input = st.text_input("Enter your sentence")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict_word(user_input)
        st.markdown(f'<div class="prediction">{result}</div>', unsafe_allow_html=True)

if st.button("Auto Complete"):
    text = user_input
    for _ in range(5):
        next_word = predict_word(text)
        text += " " + next_word

    st.success(text)

st.markdown('</div>', unsafe_allow_html=True)