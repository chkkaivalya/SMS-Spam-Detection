import streamlit as st
import pickle

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your message:")

if st.button("Predict"):
    # Preprocess and transform input
    transformed = vectorizer.transform([input_sms])
    prediction = model.predict(transformed)[0]

    if prediction == 1:
        st.header("Spam ❌")
    else:
        st.header("Not Spam ✅")

