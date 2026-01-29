import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import nltk
import warnings
import pandas as pd

warnings.simplefilter("ignore")

nltk.download("stopwords")

genre_mapping = {
    0: 'Fantasy',
    1: 'Science Fiction',
    2: 'Crime Fiction',
    3: 'Historical novel',
    4: 'Horror',
    5: 'Thriller'
}

with open('Finalized-Model.pickle', 'rb') as file:
    model, vectorizer = pickle.load(file)

df = pd.read_csv('traineddata.csv')

summary_to_book = df.set_index('summary')['book_name'].to_dict()

def cleantext(text):
    text = re.sub("'\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    text = text.lower()
    return text

stop_words = set(stopwords.words("english"))

def remove_stop_words(text):
    filtered_words = [word for word in text.split() if word not in stop_words]
    return " ".join(filtered_words)

def preprocess_summary(summary):
    summary = cleantext(summary)
    summary = remove_stop_words(summary)
    return summary

def predict_genre_and_book(summary):
    processed_summary = preprocess_summary(summary)
    summary_vectorized = vectorizer.transform([processed_summary])
    prediction = model.predict(summary_vectorized)[0]
    book_name = summary_to_book.get(summary, "Unknown Book")
    return prediction, book_name

def predict_genre(summary):
    processed_summary = preprocess_summary(summary)
    summary_vectorized = vectorizer.transform([processed_summary])
    prediction = model.predict(summary_vectorized)
    genre_index = prediction[0]
    return genre_mapping.get(genre_index, "Unknown Genre")

st.title("Books Genre Predictor")

summary_input = st.text_area("**Enter The Book Summary:**")

if st.button("Predict Genre"):
    if summary_input:
        genre_code, book_name = predict_genre_and_book(summary_input)
        genre = genre_mapping.get(genre_code, "Unknown Genre")
        st.success(f"The Predicted Genre Is: **{genre}**")
        st.info(f"The Book Name Is: **{book_name}**")
    else:
        st.error("**Please Enter A Book Summary To Get A Prediction.**")

