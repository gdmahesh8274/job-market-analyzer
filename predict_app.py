import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# UI
st.title("Job Experience Level Predictor")
st.write("Enter a job title to predict the expected experience level.")

# Input
job_title = st.text_input("Enter job title:")

# Predict
if st.button("Predict"):
    if job_title:
        input_vec = vectorizer.transform([job_title])
        prediction = model.predict(input_vec)

        label_map = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior Level', 'EX': 'Executive Level'}
        st.success(f"Predicted Experience Level: **{label_map.get(prediction[0], prediction[0])}**")
    else:
        st.warning("Please enter a job title.")
