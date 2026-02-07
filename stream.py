import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie
# stream.py (add near top)
import pandas as pd
from forecastutils import forecast_7days_iterative
# import your model objects from model.py
from model import knn_reg, knn_clf, scaler, rain_le


# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="SkyCast Analytics",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ASSET LOADING (Safe Version) ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5) # Added timeout to prevent hanging
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

# Professional Weather Animations
# Using reliable URLs. If these fail, the code below will now handle it safely.
lottie_storm = load_lottieurl("https://lottie.host/020222d1-0814-469b-980b-936c5353cc57/6qJqXjX12f.json")
lottie_sun = load_lottieurl("https://lottie.host/86d6b5e0-47b2-4d01-a128-444f23b20e06/2a2e4e2079.json")
lottie_cloud = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_kljxfhzi.json") 

# --- 3. CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    
    .stApp {
       background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20240104/pngtree-aesthetic-island-captivating-outdoor-clouds-and-sunlight-space-texture-image_13808024.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    div[data-testid="stForm"], div.css-1r6slb0 {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        padding: 2rem;
    }

    h1 { color: #1a202c; font-weight: 700; letter-spacing: -1px; }
    h3 { color: #4a5568; font-weight: 400; }
    p, label { color: #2d3748 !important; font-weight: 500; }

    .stButton>button {
        background: linear-gradient(90deg, #3182ce 0%, #63b3ed 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        # Ensure these files are in the same folder as this script
        knn_reg = pickle.load(open('knn_reg.pkl', 'rb'))
        knn_clf = pickle.load(open('knn_clf.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        rain_le = pickle.load(open('rain_le.pkl', 'rb'))
        return knn_reg, knn_clf, scaler, rain_le
    except FileNotFoundError:
        return None, None, None, None

knn_reg, knn_clf, scaler, rain_le = load_models()

# --- 5. SIDEBAR ---
with st.sidebar:
    # SAFE ANIMATION CHECK: Only try to show it if it downloaded successfully
    if lottie_cloud:
        st_lottie(lottie_cloud, height=150, key="logo_anim")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/1163/1163661.png", width=100)
        
    st.markdown("### ⚙️ System Settings")
    st.info("Ensure model files (.pkl) are in the directory.")
    st.markdown("---")
    st.write("Using K-Nearest Neighbors (KNN) for predictive analysis.")

# --- 6. MAIN INTERFACE ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("SkyCast Analytics")
    st.markdown("### Intelligent Weather Forecasting System")

if knn_reg is None:
    st.error("⚠️ Model files not found. Please run 'model.py' first.")
else:
    # --- INPUT FORM ---
    with st.form("weather_form"):
        st.markdown("#### 📡 Atmospheric Parameters")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            min_temp = st.number_input("Min Temp (°C)", value=14.0, step=0.1)
        with c2:
            temp_9am = st.number_input("Temp at 9am (°C)", value=16.0, step=0.1)
        with c3:
            pressure = st.number_input("Pressure (hPa)", value=1015.0, step=1.0)
            
        c4, c5 = st.columns([1, 2])
        with c4:
            wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        with c5:
            humidity = st.slider("Humidity at 9am (%)", 0, 100, 60)

        st.markdown("---")
        submit = st.form_submit_button("RUN PREDICTION")

    # --- RESULTS ---
    if submit:
        features = [min_temp, humidity, pressure, wind_speed, temp_9am]
        X_input = scaler.transform([features])
        
        temp_pred = knn_reg.predict(X_input)[0]
        rain_idx = knn_clf.predict(X_input)[0]
        rain_pred = rain_le.inverse_transform([rain_idx])[0]
        
        st.markdown("### 📊 Forecast Analysis")
        
        res_c1, res_c2, res_c3 = st.columns([1.5, 1.5, 1])
        
        with res_c1:
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #3182ce;">
                <p style="margin:0; font-size: 14px; color:#718096;">PREDICTED MAX TEMP</p>
                <h2 style="margin:0; font-size: 32px; color:#2d3748;">{temp_pred:.1f}°C</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with res_c2:
            border_color = "#e53e3e" if rain_pred == "Yes" else "#48bb78"
            rain_text = "High Probability" if rain_pred == "Yes" else "Low Probability"
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid {border_color};">
                <p style="margin:0; font-size: 14px; color:#718096;">RAIN FORECAST</p>
                <h2 style="margin:0; font-size: 32px; color:#2d3748;">{rain_pred}</h2>
                <p style="margin:0; font-size: 12px; color:{border_color}; font-weight:bold;">{rain_text}</p>
            </div>
            """, unsafe_allow_html=True)

        with res_c3:
            # SAFE ANIMATION CHECK FOR RESULTS
            if rain_pred == "Yes":
                if lottie_storm:
                    st_lottie(lottie_storm, height=120, key="storm")
                else:
                    st.error("🌧️ Storm Warning")
            else:
                if lottie_sun:
                    st_lottie(lottie_sun, height=120, key="sun")
                else:
                    st.success("☀️ Sunny Day")
    # --- 7-DAY ITERATIVE FORECAST (ADDED) ---
# This uses the same input variables you already collect in the form:
# min_temp, humidity, pressure, wind_speed, temp_9am
# (They exist because your form defines them with default values.)

if st.button("RUN 7-DAY FORECAST"):
    try:
        start_features = [min_temp, humidity, pressure, wind_speed, temp_9am]
        df7 = forecast_7days_iterative(knn_reg, knn_clf, scaler, rain_le, start_features, days=7)

        st.markdown("### 📅 7-Day Forecast (iterative)")
        st.dataframe(df7)              # interactive table for the 7-day forecast
        st.line_chart(df7.set_index('day')['PredMaxTemp'])  # simple chart of predicted max temps
    except Exception as e:
        st.error(f"Error producing 7-day forecast: {e}")

            
