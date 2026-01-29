# weather_prediction_app.py
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🌤️ Weather Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced ML-Based Weather Forecasting</div>', unsafe_allow_html=True)


@st.cache_resource
def load_and_train_model():
    """Load pre-trained model and preprocessing assets from disk"""
    try:
        # Define the path to your models folder
        # Note: If your app and models are in the same folder, use 'models/...' 
        # If the app is in a subfolder, use '../models/...'
        model_path = '../models/weather_model.pkl'
        scaler_path = '../models/scaler.pkl'
        le_path = '../models/label_encoder.pkl'
        
        # Load the saved components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        
        return model, scaler, le
    
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        st.info("Check that your .pkl files are in the 'models' directory.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None, None, None

# Load Model
with st.spinner('Loading ML Model... Please wait'):
    model, scaler, label_encoder = load_and_train_model()

if model is None:
    st.error("Failed to load the model. Please check your dataset path.")
    st.stop()

# Sidebar for Input
st.sidebar.header("📊 Enter Weather Parameters")

# Current Date/Time
now = datetime.now()
current_hour = now.hour
current_month = now.month

st.sidebar.markdown("### 🌡️ Temperature")
temp = st.sidebar.slider(
    "Temperature (°C)", 
    min_value=-40.0, 
    max_value=50.0, 
    value=20.0, 
    step=0.5,
    help="Current temperature in Celsius"
)

st.sidebar.markdown("### 💧 Humidity")
humidity = st.sidebar.slider(
    "Relative Humidity (%)", 
    min_value=0, 
    max_value=100, 
    value=50,
    help="Percentage of moisture in the air"
)

st.sidebar.markdown("### 💨 Wind Speed")
wind_speed = st.sidebar.slider(
    "Wind Speed (km/h)", 
    min_value=0, 
    max_value=100, 
    value=10,
    help="Current wind speed"
)

st.sidebar.markdown("### 👁️ Visibility")
visibility = st.sidebar.slider(
    "Visibility (km)", 
    min_value=0.0, 
    max_value=50.0, 
    value=10.0, 
    step=0.1,
    help="How far you can see"
)

st.sidebar.markdown("### 🌐 Atmospheric Pressure")
pressure = st.sidebar.slider(
    "Pressure (kPa)", 
    min_value=95.0, 
    max_value=105.0, 
    value=101.0, 
    step=0.1,
    help="Atmospheric pressure"
)

st.sidebar.markdown("### 🕐 Time Parameters")
hour = st.sidebar.selectbox(
    "Hour of Day", 
    range(0, 24), 
    index=current_hour,
    help="Current hour (0-23)"
)

month = st.sidebar.selectbox(
    "Month", 
    range(1, 13), 
    index=current_month-1,
    format_func=lambda x: datetime(2000, x, 1).strftime('%B'),
    help="Current month"
)

# Prediction Button
predict_button = st.sidebar.button("🔮 Predict Weather", use_container_width=True)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    if predict_button:
        # Prepare input data
        input_data = np.array([[temp, humidity, wind_speed, visibility, pressure, hour, month]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        weather_classes = ['Bad Weather', 'Clear', 'Cloudy']
        predicted_weather = weather_classes[prediction]
        
        # Weather emoji mapping
        weather_emoji = {
            'Bad Weather': '⛈️',
            'Clear': '☀️',
            'Cloudy': '☁️'
        }
        
        # Display Prediction
        st.markdown(f"""
            <div class="prediction-box">
                {weather_emoji[predicted_weather]} Predicted Weather: {predicted_weather}
            </div>
        """, unsafe_allow_html=True)
        
        # Confidence Score
        st.subheader("📊 Prediction Confidence")
        confidence_df = pd.DataFrame({
            'Weather Condition': weather_classes,
            'Probability': prediction_proba * 100
        })
        
        fig = px.bar(
            confidence_df, 
            x='Weather Condition', 
            y='Probability',
            color='Probability',
            color_continuous_scale='viridis',
            text='Probability'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            title="Confidence Levels for Each Weather Type",
            xaxis_title="Weather Condition",
            yaxis_title="Probability (%)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather Interpretation
        st.subheader("🌦️ Weather Interpretation")
        
        if predicted_weather == 'Clear':
            st.success("""
            **Clear Weather Conditions:**
            - ☀️ Expect sunny and clear skies
            - Ideal for outdoor activities
            - Low chance of precipitation
            """)
        elif predicted_weather == 'Cloudy':
            st.info("""
            **Cloudy Weather Conditions:**
            - ☁️ Expect overcast skies
            - Moderate outdoor conditions
            - Possible light precipitation
            """)
        else:
            st.warning("""
            **Bad Weather Conditions:**
            - ⛈️ Expect adverse weather
            - Rain, snow, or storms possible
            - Exercise caution for outdoor activities
            """)

with col2:
    st.subheader("📋 Input Summary")
    
    input_summary = {
        "Temperature": f"{temp}°C",
        "Humidity": f"{humidity}%",
        "Wind Speed": f"{wind_speed} km/h",
        "Visibility": f"{visibility} km",
        "Pressure": f"{pressure} kPa",
        "Hour": f"{hour}:00",
        "Month": datetime(2000, month, 1).strftime('%B')
    }
    
    for key, value in input_summary.items():
        st.markdown(f"""
            <div class="info-box">
                <strong>{key}:</strong> {value}
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 Powered by XGBoost Machine Learning Algorithm</p>
        <p>Built with ❤️ using Streamlit | Data Science Project 2025</p>
    </div>
""", unsafe_allow_html=True)

# Additional Information Section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### How This Works:
    
    This weather prediction system uses **XGBoost**, a powerful machine learning algorithm, 
    to analyze historical weather patterns and predict current conditions.
    
    **Key Features:**
    - 🎯 81%+ Accuracy
    - ⚡ Real-time predictions
    - 📊 Confidence scoring
    - 🌍 Multi-parameter analysis
    
    **Input Parameters:**
    1. **Temperature**: Current air temperature
    2. **Humidity**: Moisture content in the air
    3. **Wind Speed**: Current wind velocity
    4. **Visibility**: How far you can see
    5. **Pressure**: Atmospheric pressure
    6. **Hour**: Time of day
    7. **Month**: Current month
    
    The model classifies weather into three categories:
    - ☀️ **Clear**: Sunny, clear skies
    - ☁️ **Cloudy**: Overcast conditions
    - ⛈️ **Bad Weather**: Rain, snow, storms, fog
    """)

with st.expander("📈 Model Performance Metrics"):
    st.markdown("""
    ### Model Accuracy: 81%
    
    | Weather Class | Precision | Recall | F1-Score |
    |--------------|-----------|--------|----------|
    | Bad Weather  | 93%       | 90%    | 91%      |
    | Clear        | 78%       | 80%    | 79%      |
    | Cloudy       | 74%       | 75%    | 74%      |
    
    **Training Data:** 8,784 hourly weather observations from 2012
    """)
