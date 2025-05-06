import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model and scaler
model = joblib.load("../output/breast_cancer_model.pkl")
scaler = joblib.load("../output/scaler.pkl")

# Define feature names
feature_names = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']

# Sample data (first two rows from the dataset for testing)
# Row 1: Cancer (0), Row 2: Healthy (1)
malignant_sample = [48, 23.5, 70, 2.707, 0.467409, 8.8071, 9.7024, 7.99585, 417.114]
benign_sample = [83, 20.69, 92, 3.115, 0.706897, 8.8438, 5.4293, 4.06405, 468.786]

# Custom CSS for styling
st.markdown("""
    <style>
    /* Import a modern font similar to GitHub's */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* Apply the font and dark gradient background */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0D1117 0%, #1F0A44 100%) !important; /* Dark blue to purple gradient */
        color: #FFFFFF; /* White text */
        min-height: 100vh; /* Ensure the gradient covers the entire viewport */
    }

    /* Title styling */
    .title {
        font-size: 56px; /* Larger like GitHub */
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Tagline styling */
    .tagline {
        font-size: 20px; /* Slightly larger like GitHub */
        font-weight: 300;
        color: #C9D1D9; /* Light gray */
        text-align: center;
        margin-bottom: 40px;
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 24px;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Input field styling */
    input[type="text"] {
        background-color: #FFFFFF !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 16px !important;
        color: #0D1117 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #238636 0%, #2EA043 100%) !important; /* Green gradient like GitHub's Sign up */
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4) !important;
    }

    /* Style for Clear and Reset buttons */
    .stButton>button[label="Clear"],
    .stButton>button[label="Reset to Default"] {
        background: linear-gradient(90deg, #30363D 0%, #21262D 100%) !important; /* Dark gradient like GitHub's Try Copilot */
        color: #FFFFFF !important;
    }

    /* Result styling */
    .result.malignant {
        font-size: 24px;
        font-weight: 600;
        color: #FF5555; /* Red for malignant */
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(255, 85, 85, 0.1) 0%, rgba(255, 205, 210, 0.1) 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }
    .result.benign {
        font-size: 24px;
        font-weight: 600;
        color: #2EA043; /* Green for benign */
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.1) 0%, rgba(200, 230, 201, 0.1) 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }
    .result.error {
        font-size: 20px;
        font-weight: 600;
        color: #FFAA33; /* Yellow for error */
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(255, 170, 51meva: 1px solid rgba(255, 170, 51, 0.1);
        background: linear-gradient(135deg, rgba(255, 245, 157, 0.1) 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Confidence score styling */
    .confidence {
        font-size: 16px;
        font-weight: 400;
        color: #C9D1D9;
        text-align: center;
        margin-top: 10px;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #238636 0%, #2EA043 100%) !important;
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Expander styling */
    .stExpander {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stExpander p {
        color: #C9D1D9 !important;
    }

    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        color: #0D1117 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div > div {
        color: #0D1117 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div > div > div {
        color: #0D1117 !important;
    }
    /* Style the dropdown menu (options list) */
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] {
        background-color: #FFFFFF !important; /* White background for dropdown menu */
        color: #0D1117 !important; /* Black text for dropdown options */
    }
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] ul {
        background-color: #FFFFFF !important; /* White background for dropdown menu */
        color: #0D1117 !important; /* Black text for dropdown options */
    }
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] ul li {
        background-color: #FFFFFF !important; /* White background for each option */
        color: #0D1117 !important; /* Black text for each option */
    }
    .stSelectbox div[data-baseweb="select"] div[role="listbox"] ul li:hover {
        background-color: #F0F0F0 !important; /* Light gray hover effect */
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #0D1117 0%, #1F0A44 100%) !important; /* Match main area's gradient */
        color: #FFFFFF !important;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg p {
        color: #C9D1D9 !important; /* Light gray text */
    }

    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #1F0A44 0%, #1F0A44 100%);
        color: #C9D1D9;
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 -4px 8px rgba(0, 0, 0, 0.2);
    }
    .footer a {
        color: #58A6FF;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        color: #FFFFFF;
        text-decoration: underline;
    }

    /* Ensure footer is at the bottom */
    html, body, [class*="css"] {
        height: 100%;
        margin: 0;
    }
    .stApp {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }
    .main-content {
        flex: 1;
    }

    /* Reduce margin for selectbox to minimize empty space */
    .stSelectbox {
        margin-bottom: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2>About BreastCancerSense</h2>", unsafe_allow_html=True)
    st.markdown("""
        BreastCancerSense is a machine learning tool designed to assist in the early detection of breast cancer using clinical data. It leverages the Breast Cancer Coimbra Dataset and an XGBoost model to predict cancer risk with high accuracy.
    """)
    st.markdown("<h3>Dataset</h3>", unsafe_allow_html=True)
    st.markdown("The dataset includes 116 samples with 9 clinical features such as Age, BMI, Glucose, and more.")
    st.markdown("<h3>Model</h3>", unsafe_allow_html=True)
    st.markdown("The model uses XGBoost with hyperparameter tuning and probability calibration for reliable predictions.")

# Main content
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

# Title and Tagline
st.markdown("<div class='title'>BreastCancerSense</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Empowering Early Detection with AI</div>", unsafe_allow_html=True)

# Description (without container)
st.markdown("""
    <div style='font-size: 16px; font-weight: 400; color: #C9D1D9; text-align: center; margin: 0 auto; max-width: 700px; margin-bottom: 40px;'>
    Welcome to <b>BreastCancerSense</b>! This advanced tool uses clinical data to predict breast cancer risk with precision. Simply enter the 9 clinical features below and click 'Predict' to get an instant result. Designed to assist in early diagnosis, this application harnesses the power of machine learning for better healthcare outcomes.
    </div>
""", unsafe_allow_html=True)

# Initialize session state for form data
if "form_data" not in st.session_state:
    st.session_state.form_data = {feature: float(benign_sample[feature_names.index(feature)]) for feature in feature_names}

# Sample data selector
sample_option = st.selectbox(
    "Select Sample Data (Optional)",
    ["None", "Cancer Sample (First Row)", "Healthy Sample (Second Row)"],
    help="Select a sample dataset to pre-fill the form with Cancer or Healthy data for testing."
)

# Update form data based on sample selection
if sample_option == "Cancer Sample (First Row)":
    st.session_state.form_data = {feature: float(malignant_sample[feature_names.index(feature)]) for feature in feature_names}
elif sample_option == "Healthy Sample (Second Row)":
    st.session_state.form_data = {feature: float(benign_sample[feature_names.index(feature)]) for feature in feature_names}

# Input fields
form_inputs = {}

# Features
st.markdown("<div class='subtitle'>Enter Clinical Features</div>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        form_inputs['Age'] = st.text_input(
            "Age",
            value=str(st.session_state.form_data['Age']),
            key="feature_0",
            help="Enter the patient's age in years (e.g., 45)"
        )
        form_inputs['BMI'] = st.text_input(
            "BMI",
            value=str(st.session_state.form_data['BMI']),
            key="feature_1",
            help="Enter the Body Mass Index in kg/m² (e.g., 25.5)"
        )
        form_inputs['Glucose'] = st.text_input(
            "Glucose",
            value=str(st.session_state.form_data['Glucose']),
            key="feature_2",
            help="Enter the Glucose level in mg/dL (e.g., 90)"
        )
    with col2:
        form_inputs['Insulin'] = st.text_input(
            "Insulin",
            value=str(st.session_state.form_data['Insulin']),
            key="feature_3",
            help="Enter the Insulin level in µIU/mL (e.g., 3.5)"
        )
        form_inputs['HOMA'] = st.text_input(
            "HOMA",
            value=str(st.session_state.form_data['HOMA']),
            key="feature_4",
            help="Enter the Homeostatic Model Assessment of Insulin Resistance (e.g., 0.7)"
        )
        form_inputs['Leptin'] = st.text_input(
            "Leptin",
            value=str(st.session_state.form_data['Leptin']),
            key="feature_5",
            help="Enter the Leptin level in ng/mL (e.g., 8.5)"
        )
    with col3:
        form_inputs['Adiponectin'] = st.text_input(
            "Adiponectin",
            value=str(st.session_state.form_data['Adiponectin']),
            key="feature_6",
            help="Enter the Adiponectin level in µg/mL (e.g., 5.4)"
        )
        form_inputs['Resistin'] = st.text_input(
            "Resistin",
            value=str(st.session_state.form_data['Resistin']),
            key="feature_7",
            help="Enter the Resistin level in ng/mL (e.g., 4.0)"
        )
        form_inputs['MCP.1'] = st.text_input(
            "MCP.1",
            value=str(st.session_state.form_data['MCP.1']),
            key="feature_8",
            help="Enter the Monocyte Chemoattractant Protein-1 level in pg/mL (e.g., 468)"
        )

# Form for buttons (centered)
with st.form(key="prediction_form"):
    button_container = st.container()
    with button_container:
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        with button_col1:
            submit_button = st.form_submit_button("Predict", help="Click to predict if the patient has breast cancer", use_container_width=True)
        with button_col2:
            clear_button = st.form_submit_button("Clear", help="Click to clear all input fields", use_container_width=True)
        with button_col3:
            reset_button = st.form_submit_button("Reset to Default", help="Click to reset to default values", use_container_width=True)

# Handle Clear button
if clear_button:
    st.session_state.form_data = {feature: "" for feature in feature_names}
    st.rerun()

# Handle Reset button
if reset_button:
    st.session_state.form_data = {feature: float(benign_sample[feature_names.index(feature)]) for feature in feature_names}
    st.rerun()

# Prediction logic
if submit_button:
    with st.spinner("Analyzing clinical data..."):
        # Validate inputs
        patient_data = []
        for feature in feature_names:
            value = form_inputs[feature]
            if not value:
                st.markdown("<div class='result error'>Error: All fields must be filled!</div>", unsafe_allow_html=True)
                st.stop()
            try:
                patient_data.append(float(value))
            except ValueError:
                st.markdown(f"<div class='result error'>Error: Invalid value for {feature}. Please enter a valid number.</div>", unsafe_allow_html=True)
                st.stop()

        # Make prediction
        patient_df = pd.DataFrame([patient_data], columns=feature_names)
        patient_df_scaled = scaler.transform(patient_df.to_numpy())
        probas = model.predict_proba(patient_df_scaled)
        threshold = 0.5
        prediction = 1 if probas[0][1] >= threshold else 0
        result = "Cancer (Malignant)" if prediction == 0 else "Healthy (Non-Cancerous)"
        result_class = "malignant" if prediction == 0 else "benign"

        # Calculate confidence score
        confidence = max(probas[0]) * 100  # Convert to percentage

        # Display result
        
        st.markdown(f"<div class='result {result_class}'>Prediction: {result}</div>", unsafe_allow_html=True)
        st.markdown("<div class='confidence'>Confidence Score</div>", unsafe_allow_html=True)
        st.progress(confidence / 100)


# Display Confusion Matrix
with st.expander("View Model Performance (Confusion Matrix)"):
    st.image("../output/prediction_counts.jpeg", caption="Confusion Matrix", use_container_width=True)

# Display Feature Importance (if available)
try:
    with st.expander("View Feature Importance"):
        st.image("../output/feature_importance.jpeg", caption="Feature Importance", use_container_width=True)
except:
    st.write("Feature Importance plot not available.")

# Close main content
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
    Developed by Diksha Gupta | Major Project | 2025<br>
    <a href='https://www.linkedin.com/in/dikshagupta22' target='_blank'>LinkedIn</a>
    </div>
""", unsafe_allow_html=True)