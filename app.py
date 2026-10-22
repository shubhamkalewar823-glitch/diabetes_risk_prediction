import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Health Risk Prediction App",
    page_icon="🩺",
    layout="wide",
)


# Load the saved model
@st.cache_resource
def load_model():
    # Replace 'model.pkl' with your exact pickle file name if different,
    # ensuring you do not change the original pkl file name itself.
    with open("svmc.pkl", "rb") as file:
        model = pickle.load(file)
    return model


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file (`model.pkl`) not found! Please place your pickle file in the same directory as `app.py`."
    )
    st.stop()

# App Header
st.title("🩺 Health Risk Prediction Dashboard")
st.markdown(
    "Provide the patient details below to get real-time classification predictions from the SVC model."
)

st.divider()

# Layout using columns for a clean interface
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Anthropometrics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    city = st.text_input("City", value="New York")
    bmi = st.number_input(
        "BMI", min_value=10.0, max_value=60.0, value=24.5, format="%.1f"
    )
    waist_circumference_cm = st.number_input(
        "Waist Circumference (cm)", min_value=40.0, max_value=200.0, value=85.0
    )
    income_bracket = st.selectbox(
        "Income Bracket", ["Low", "Medium", "High", "Very High"]
    )

with col2:
    st.subheader("Lifestyle & Habits")
    family_history_diabetes = st.selectbox(
        "Family History of Diabetes", ["No", "Yes"]
    )
    physical_activity_level = st.selectbox(
        "Physical Activity Level", ["Low", "Moderate", "High"]
    )
    diet_type = st.selectbox(
        "Diet Type", ["Standard", "Vegetarian", "Vegan", "Keto", "Mediterranean"]
    )
    smoking_status = st.selectbox(
        "Smoking Status", ["Never", "Former", "Current"]
    )
    alcohol_consumption = st.selectbox(
        "Alcohol Consumption", ["None", "Low", "Moderate", "High"]
    )

with col3:
    st.subheader("Clinical & Vital Measurements")
    hours_sleep_per_night = st.slider(
        "Hours Sleep Per Night", min_value=1.0, max_value=14.0, value=7.0
    )
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
    fasting_blood_sugar = st.number_input(
        "Fasting Blood Sugar", min_value=50.0, max_value=300.0, value=100.0
    )
    hba1c_level = st.number_input(
        "HbA1c Level", min_value=3.0, max_value=15.0, value=5.5, format="%.1f"
    )
    blood_pressure_systolic = st.number_input(
        "Blood Pressure - Systolic", min_value=70, max_value=200, value=120
    )
    blood_pressure_diastolic = st.number_input(
        "Blood Pressure - Diastolic", min_value=40, max_value=130, value=80
    )

st.divider()

# Prediction Button
if st.button("Run Prediction", type="primary", use_container_width=True):
    # Construct input dataframe matching the expected features
    # Note: Depending on how your model was trained (e.g., if it uses OneHotEncoding),
    # ensure input formatting aligns with the pipeline pipeline/encoder requirements.
    input_data = pd.DataFrame(
        {
            "age": [age],
            "gender": [gender],
            "city": [city],
            "bmi": [bmi],
            "family_history_diabetes": [family_history_diabetes],
            "physical_activity_level": [physical_activity_level],
            "diet_type": [diet_type],
            "smoking_status": [smoking_status],
            "alcohol_consumption": [alcohol_consumption],
            "hours_sleep_per_night": [hours_sleep_per_night],
            "stress_level": [stress_level],
            "fasting_blood_sugar": [fasting_blood_sugar],
            "hba1c_level": [hba1c_level],
            "blood_pressure_systolic": [blood_pressure_systolic],
            "blood_pressure_diastolic": [blood_pressure_diastolic],
            "waist_circumference_cm": [waist_circumference_cm],
            "income_bracket": [income_bracket],
        }
    )

    try:
        prediction = model.predict(input_data)
        st.success(f"### Prediction Result: **{prediction[0]}**")

        # If probability estimation is enabled in your SVC model
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)
            st.info(
                f"Prediction Confidence / Probabilities: `{np.max(proba)*100:.2f}%`"
            )
    except Exception as e:
        st.error(
            f"Error during prediction. Make sure categorical features match training encoding: {e}"
        )
