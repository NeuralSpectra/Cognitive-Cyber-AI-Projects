import streamlit as st
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import warnings

warnings.simplefilter("ignore")

with open("Tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)

model = load_model("Finalized-Model.h5")

st.title("Advertisement Classifier")

title_input = st.text_input("**Enter The Title:**")
description_input = st.text_area("**Enter The Description:**")

def predict_category(title, description):

    combined_text = title + " " + description
    sequence = tokenizer.texts_to_sequences([combined_text])
    padded_sequence = pad_sequences(sequence, maxlen=50)
    prediction = model.predict(padded_sequence)
    predicted_category = prediction.argmax(axis=-1)[0]

    category_mapping = {
        0: 'Art and Music',
        1: 'Food',
        2: 'History',
        3: 'Manufacturing',
        4: 'Science and Technology',
        5: 'Travel'
    }

    return category_mapping[predicted_category]

if st.button("Predict Category"):
    if title_input and description_input:
        predicted_category = predict_category(title_input, description_input)
        st.success(f"The Predicted Category Is: **{predicted_category}**")
    else:
        st.error("**Please Provide Both Title And Description.**")

