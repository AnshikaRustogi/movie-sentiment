import streamlit as st
import joblib
from preprocess import clean_text

model = joblib.load('src/models/model.pkl')
vectorizer = joblib.load('src/models/vectorizer.pkl')

st.title("Movie Sentiment Analysis")

review = st.text_area("Enter your review:")

if st.button("Analyze"):
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]

    if result == 1:
        st.success("Positive")
    else:
        st.error("Negative")
