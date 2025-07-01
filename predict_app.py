import streamlit as st
import joblib
import pandas as pd
import io
from PIL import Image



# Set page config (must be FIRST Streamlit command)
st.set_page_config(page_title="Job Experience Predictor", page_icon="🧳", layout="centered")

# Logo
logo = Image.open("logo.png")  # make sure the file exists
st.image(logo, width=150)

# Header
st.markdown("<h1 style='text-align: center;'>🧳 Job Experience Level Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use AI to predict experience level from job title</p>", unsafe_allow_html=True)

# Load models
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
salary_model = joblib.load("salary_model.pkl")
salary_vectorizer = joblib.load("salary_vectorizer.pkl")

# Label map
label_map = {
    0: "Junior Level",
    1: "Mid Level",
    2: "Senior Level"
}

# Extract skills
def extract_skills(title):
    skills = ["python", "sql", "aws", "excel", "tableau", "power bi", "spark", "pandas", "numpy"]
    detected = [skill for skill in skills if skill.lower() in title.lower()]
    return detected
# Input
#theme = st.radio("🌓 Choose Theme:", ["Light", "Dark"], horizontal=True)
st.write("🔍 **Enter a Job Title Below:**")
job_title = st.text_input("")

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear"):
        st.session_state["job_title"] = ""
        st.rerun()


with col2:
    if st.button("✨ Predict Experience Level"):
        if job_title:
            # Predict Experience
            input_vec = vectorizer.transform([job_title])
            prediction = model.predict(input_vec)

            # Extract skills
            detected_skills = extract_skills(job_title)

            # Salary
            salary_vec = salary_vectorizer.transform([job_title])
            salary_prediction = salary_model.predict(salary_vec)

            # Output
            st.success(f"🎯 **Predicted:** 🟠 {label_map.get(prediction[0], prediction[0])}")
            st.info(f"🌸 **Likely Skills:** {', '.join(detected_skills) if detected_skills else 'None'}")
            st.warning(f"💰 **Estimated Salary Range:** **{salary_prediction[0]}**")

            # Export results
            result_df = pd.DataFrame({
                "Job Title": [job_title],
                "Experience Level": [label_map.get(prediction[0], prediction[0])],
                "Likely Skills": [", ".join(detected_skills) if detected_skills else "None"],
                "Salary Range": [salary_prediction[0]]
            })

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "job_result.csv", "text/csv")

            pdf_buffer = io.BytesIO()
            result_df.to_csv(pdf_buffer, index=False)
            st.download_button("⬇️ Download PDF (simulated)", pdf_buffer, "job_result.pdf")
