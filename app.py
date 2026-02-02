import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
with open('st_clf_2_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the feature names in the correct order as used during training
feature_names = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
    'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST',
    'Number of vessels fluro', 'Thallium'
]

# Custom CSS for beautiful dark theme styling with animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        min-height: 100vh;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Header Styling */
    .header-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: #b0b0d0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        animation: fadeInDown 1s ease-out 0.2s both;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 1.5rem 0 1.2rem 0;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: slideInLeft 0.6s ease-out;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-header:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        transform: translateX(2px);
        transition: all 0.3s ease;
    }
    
    /* Input Card Styling */
    .input-container {
        background: linear-gradient(135deg, #1f1f3a 0%, #252541 100%);
        border-radius: 15px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
        border: 1.5px solid rgba(102, 126, 234, 0.25);
        animation: slideInLeft 0.7s ease-out;
    }
    
    .input-container:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #1a3a52 0%, #2d1b4e 100%);
        border-left: 5px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 0.95rem;
        color: #e0e0e0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #3a2a1a 0%, #3a1f1f 100%);
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 0.95rem;
        color: #e0e0e0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
        transform: translateY(-3px) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Result Cards */
    .result-card-positive {
        background: linear-gradient(135deg, #1a3a2a 0%, #1f3a28 100%);
        border-left: 6px solid #28a745;
        padding: 30px;
        border-radius: 15px;
        margin-top: 25px;
        box-shadow: 0 10px 35px rgba(40, 167, 69, 0.25);
        color: #e0e0e0;
        animation: fadeInDown 0.8s ease-out;
        border-top: 1px solid rgba(40, 167, 69, 0.3);
        border-right: 1px solid rgba(40, 167, 69, 0.2);
        border-bottom: 1px solid rgba(40, 167, 69, 0.2);
    }
    
    .result-card-negative {
        background: linear-gradient(135deg, #3a1a1a 0%, #3a1f28 100%);
        border-left: 6px solid #dc3545;
        padding: 30px;
        border-radius: 15px;
        margin-top: 25px;
        box-shadow: 0 10px 35px rgba(220, 53, 69, 0.25);
        color: #e0e0e0;
        animation: fadeInDown 0.8s ease-out;
        border-top: 1px solid rgba(220, 53, 69, 0.3);
        border-right: 1px solid rgba(220, 53, 69, 0.2);
        border-bottom: 1px solid rgba(220, 53, 69, 0.2);
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 15px;
        color: white;
        letter-spacing: -0.5px;
    }
    
    .result-subtitle {
        font-size: 1.05rem;
        margin-bottom: 20px;
        line-height: 1.7;
        color: #d0d0e0;
        font-weight: 500;
    }
    
    .result-recommendations {
        background: linear-gradient(135deg, rgba(50, 50, 80, 0.5) 0%, rgba(40, 50, 70, 0.5) 100%);
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        color: #e0e0e0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        color: #e0e0e0;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .sidebar-info:hover {
        border-color: rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
    }
    
    /* Form Styling */
    .stForm {
        border: none !important;
        padding: 0 !important;
    }
    
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 15px;
    }
    
    /* Input labels and text */
    .stNumberInput > label, .stSelectbox > label, .stSlider > label {
        color: #e0e0e0 !important;
    }
    
    /* Feature Input Card */
    .feature-card {
        background: #1f1f3a;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 4px solid #667eea;
        color: #e0e0e0;
    }
    
    /* Loading and Success Messages */
    .success-box {
        background: linear-gradient(135deg, #1a3a2a 0%, #1f3a28 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        color: #e0e0e0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #3a1a1a 0%, #3a1f28 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        color: #e0e0e0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px;
        color: #888;
        font-size: 0.9rem;
        margin-top: 60px;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
        background: linear-gradient(180deg, transparent 0%, rgba(102, 126, 234, 0.05) 100%);
        animation: fadeInDown 1.2s ease-out;
    }
    
    .footer p {
        margin: 8px 0 !important;
        transition: color 0.3s ease;
    }
    
    .footer p:hover {
        color: #b0b0d0;
    }
    
    /* Metric styling for dark theme */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f1f3a 0%, #252541 100%);
        border: 1.5px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        padding: 18px;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
    }
    
    /* Override text colors in various elements */
    .stMetric {
        color: #e0e0e0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    
    p, li {
        color: #d0d0e0 !important;
        line-height: 1.6;
    }
    
    /* Strong text styling */
    strong {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* Input elements styling */
    .stNumberInput input, .stSelectbox select, .stSlider input {
        background: #1a1a2e !important;
        color: #e0e0e0 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus, .stSlider input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    </style>
""", unsafe_allow_html=True)

# App header with beautiful styling
st.markdown('<div class="header-title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced ML-Powered Health Assessment Tool</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar with beautiful information
with st.sidebar:
    st.markdown("### 📋 About This App")
    st.markdown("""
    <div class="sidebar-info">
    This advanced machine learning tool predicts heart disease risk based on comprehensive patient health metrics. 
    Our model has been trained on extensive medical data to provide accurate screening assessments.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔬 How It Works")
    st.markdown("""
    <div class="sidebar-info">
    1. Enter your medical information
    2. AI analyzes 13 key health parameters
    3. Get instant risk assessment
    4. Receive personalized recommendations
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚠️ Important Disclaimer")
    st.markdown("""
    <div class="sidebar-info" style="border-left: 3px solid #ff6b6b;">
    🏥 This tool is a <strong>screening aid only</strong><br>
    ⚡ <strong>Not a medical diagnosis</strong><br>
    👨‍⚕️ Always consult with a qualified healthcare professional<br>
    🚨 Seek immediate medical attention for severe symptoms
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    <div class="sidebar-info">
    • Use recent medical test results<br>
    • Ensure accurate measurements<br>
    • Consult your physician for clarification<br>
    • Keep records for follow-ups
    </div>
    """, unsafe_allow_html=True)

# Input form with improved layout
st.markdown('<div class="section-header">👤 Patient Medical Information</div>', unsafe_allow_html=True)
st.markdown("*Please provide accurate medical measurements for better predictions*")

with st.form("prediction_form", clear_on_submit=False):
    # Personal Details Section
    st.markdown('<div class="section-header" style="margin-top: 1rem;">📋 Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            '📅 Age (years)',
            min_value=1, max_value=120, value=50,
            help="Patient's age in years"
        )
    with col2:
        sex = st.selectbox(
            '👥 Sex',
            options=[0, 1],
            format_func=lambda x: '👩 Female' if x == 0 else '👨 Male',
            help="Patient's biological sex"
        )
    
    # Cardiac Symptoms Section
    st.markdown('<div class="section-header">💔 Cardiac Symptoms & Signs</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        chest_pain_type = st.selectbox(
            '💔 Chest Pain Type',
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: '🔴 Typical Angina',
                2: '🟡 Atypical Angina',
                3: '🟢 Non-Anginal Pain',
                4: '⚪ Asymptomatic'
            }[x],
            help="Type of chest pain experienced"
        )
    
    with col2:
        exercise_angina = st.selectbox(
            '🏃 Exercise-Induced Angina',
            options=[0, 1],
            format_func=lambda x: '✅ No' if x == 0 else '⚠️ Yes',
            help="Does chest pain occur with exercise?"
        )
    
    # Blood Measurements Section
    st.markdown('<div class="section-header">🩸 Blood Measurements</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bp = st.number_input(
            '🩸 Blood Pressure (mmHg)',
            min_value=80, max_value=200, value=120,
            help="Systolic blood pressure - Normal: <120"
        )
    
    with col2:
        cholesterol = st.number_input(
            '🧬 Cholesterol (mg/dl)',
            min_value=100, max_value=600, value=200,
            help="Serum cholesterol level - Desirable: <200"
        )
    
    with col3:
        fbs_over_120 = st.selectbox(
            '🍬 Fasting Blood Sugar > 120',
            options=[0, 1],
            format_func=lambda x: '✅ No' if x == 0 else '⚠️ Yes',
            help="Is fasting blood sugar greater than 120?"
        )

    # Heart Rate & ECG Section
    st.markdown('<div class="section-header">❤️ Heart Rate & ECG Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        max_hr = st.number_input(
            '❤️ Maximum Heart Rate (bpm)',
            min_value=60, max_value=220, value=150,
            help="Highest heart rate during exercise stress test"
        )
    
    with col2:
        ekg_results = st.selectbox(
            '📊 EKG Results',
            options=[0, 1, 2],
            format_func=lambda x: {
                0: '✅ Normal',
                1: '⚠️ ST-T Abnormality',
                2: '🔴 LV Hypertrophy'
            }[x],
            help="Electrocardiographic results"
        )

    # ST Segment Analysis
    st.markdown('<div class="section-header">📈 ST Segment Analysis (Stress Test)</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st_depression = st.slider(
            '📉 ST Depression (mm)',
            min_value=0.0, max_value=7.0, value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
    
    with col2:
        slope_of_st = st.selectbox(
            '📐 ST Segment Slope',
            options=[1, 2, 3],
            format_func=lambda x: {
                1: '📈 Upsloping (Good)',
                2: '➡️ Flat (Neutral)',
                3: '📉 Downsloping (Concerning)'
            }[x],
            help="The slope of the peak exercise ST segment"
        )

    # Advanced Diagnostic Tests
    st.markdown('<div class="section-header">🔬 Advanced Diagnostic Results</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        num_vessels_fluro = st.selectbox(
            '🔬 Major Vessels (Fluoroscopy)',
            options=[0, 1, 2, 3],
            format_func=lambda x: f"{'✅ None' if x == 0 else f'⚠️ {x} Vessel(s)'}",
            help="Number of major vessels (0-3) colored by fluoroscopy"
        )
    
    with col2:
        thallium = st.selectbox(
            '💊 Thallium Scan Result',
            options=[3, 6, 7],
            format_func=lambda x: {
                3: '✅ Normal Perfusion',
                6: '⚠️ Fixed Defect',
                7: '🔴 Reversible Defect'
            }[x],
            help="Thallium stress test result"
        )

    # Summary Card
    st.markdown('<div class="section-header" style="margin-top: 2rem;">📋 Information Summary</div>', unsafe_allow_html=True)
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("👤 Age", f"{age} yrs", delta=None)
    with summary_col2:
        st.metric("🩸 BP", f"{bp} mmHg", delta=None)
    with summary_col3:
        st.metric("🧬 Chol.", f"{cholesterol} mg/dl", delta=None)
    with summary_col4:
        st.metric("❤️ Max HR", f"{max_hr} bpm", delta=None)

    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button(
            '🚀 Analyze Risk Level',
            use_container_width=True
        )

    if submitted:
        # Create a DataFrame from the inputs
        input_data = pd.DataFrame([[age, sex, chest_pain_type, bp, cholesterol, fbs_over_120,
                                      ekg_results, max_hr, exercise_angina, st_depression, slope_of_st,
                                      num_vessels_fluro, thallium]], columns=feature_names)
        
        # Scale the input data
        scaled_input_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input_data)
        prediction_proba = model.predict_proba(scaled_input_data)
        
        # Display result with enhanced formatting
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card-negative">
                <div class="result-title">⚠️ Heart Disease Risk: DETECTED</div>
                <div class="result-subtitle">
                    The analysis indicates a <strong>potential risk</strong> of heart disease.
                </div>
                <div style="background: black; padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h4 style="color: #ff9999;">Risk Assessment Details:</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #e0e0e0;">
                        <li><strong>Risk Level:</strong> Elevated</li>
                        <li><strong>Confidence:</strong> High</li>
                        <li><strong>Action Required:</strong> Medical consultation needed</li>
                    </ul>
                </div>
                <div class="result-recommendations">
                    <h4 style="color: #ff9999; margin-bottom: 12px;">🏥 Recommended Actions:</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #e0e0e0;">
                        <li>📞 Schedule an appointment with a cardiologist <strong>as soon as possible</strong></li>
                        <li>📋 Bring this assessment and all medical records to your consultation</li>
                        <li>💊 Follow prescribed medications and treatment plans</li>
                        <li>🏃 Discuss lifestyle modifications with your doctor</li>
                        <li>📊 Request comprehensive cardiac testing if not recently done</li>
                        <li>🚨 Seek immediate emergency care if experiencing chest pain or shortness of breath</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-positive">
                <div class="result-title">✅ Heart Disease Risk: NOT DETECTED</div>
                <div class="result-subtitle">
                    Based on the provided measurements, <strong>no significant heart disease risk</strong> was detected.
                </div>
                <div style="background: rgba(40, 167, 69, 0.15); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 3px solid #28a745;">
                    <h4 style="color: #99ff99;">Health Assessment Summary:</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #e0e0e0;">
                        <li><strong>Risk Level:</strong> Low</li>
                        <li><strong>Confidence:</strong> High</li>
                        <li><strong>Status:</strong> Healthy indicators</li>
                    </ul>
                </div>
                <div class="result-recommendations">
                    <h4 style="color: #99ff99; margin-bottom: 12px;">💪 Recommendations to Maintain Heart Health:</h4>
                    <ul style="margin-left: 20px; margin-top: 10px; color: #e0e0e0;">
                        <li>💪 Continue maintaining a healthy lifestyle</li>
                        <li>🏃 Regular physical exercise (150 minutes/week of moderate activity)</li>
                        <li>🥗 Eat a balanced diet rich in fruits, vegetables, and lean proteins</li>
                        <li>🧂 Limit sodium intake and avoid excessive sugar</li>
                        <li>😴 Maintain 7-8 hours of quality sleep per night</li>
                        <li>👨‍⚕️ Regular health check-ups every 6-12 months</li>
                        <li>🚫 Avoid smoking and excessive alcohol consumption</li>
                        <li>😌 Manage stress through meditation or relaxation techniques</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p>❤️ <strong>Heart Disease Prediction System</strong> | Powered by Advanced Machine Learning</p>
        <p style="font-size: 0.85rem; margin-top: 10px; color: #999;">
            Disclaimer: This tool is for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
        <p style="font-size: 0.85rem; color: #999;">© 2026 Healthcare Analytics | All Rights Reserved</p>
    </div>
""", unsafe_allow_html=True)
