import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from scipy.sparse import hstack
from src.features import get_manual_features
from src.preprocessing import clean_text

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="AutoJudge",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS (Glassmorphism & Modern UI) ---
st.markdown("""
<style>
    /* 1. Dynamic Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(16, 20, 40) 0%, rgb(5, 5, 10) 90%);
        color: #e0e0e0;
    }

    /* 2. Glassmorphic Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
    }

    /* 3. Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }

    /* 4. Inputs (White Text, Clean Look) */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #3a7bd5;
        background-color: rgba(255, 255, 255, 0.08);
    }

    /* 5. Modern Button */
    .stButton > button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
    }

    /* 6. Custom Cards for Metrics */
    div[data-testid="stMetricValue"] {
        background: -webkit-linear-gradient(0deg, #00d2ff, #928DAB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0;
        font-size: 14px;
    }

    /* 7. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 20, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        clf = joblib.load('models/classifier_model.pkl')
        reg = joblib.load('models/regressor_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        return clf, reg, tfidf
    except FileNotFoundError:
        st.error("System Error: Models not found. Run main.py first.")
        st.stop()


clf_model, reg_model, tfidf_vectorizer = load_assets()

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.markdown("## AutoJudge")
    st.markdown("---")

    st.markdown("### System Status")
    st.success("Model Engine: Online")
    st.info("Vectorizer: Ready")
    st.markdown("---")

    st.markdown("### Configuration")
    st.checkbox("Math Symbol Detection", value=True, disabled=True)
    st.checkbox("Keyword Heuristics", value=True, disabled=True)
    st.markdown("---")

    with st.expander("About"):
        st.write("This tool utilizes a hybrid RF-SVM architecture to analyze algorithmic complexity.")

# --- 5. MAIN UI ---
col_logo, col_title = st.columns([1, 10])
with col_title:
    st.title("AutoJudge")
    st.markdown("Advanced difficulty prediction for algorithmic problems.")

st.markdown("---")

# Layout: Input on Left, Live Stats on Right
col_input, col_context = st.columns([2, 1])

with col_input:
    st.subheader("Input Parameters")
    title = st.text_input("Problem Title", placeholder="Enter problem name...")
    desc = st.text_area("Problem Statement", height=250, placeholder="Paste the full problem description here...")

    c1, c2 = st.columns(2)
    with c1:
        in_desc = st.text_area("Input Constraints", height=100)
    with c2:
        out_desc = st.text_area("Output Format", height=100)

    analyze_btn = st.button("Run Complexity Analysis")

with col_context:
    st.subheader("Live Monitor")

    if desc:
        l = len(desc)
        w = len(desc.split())
        st.markdown(f"""
        <div style="padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px; border: 1px solid rgba(255,255,255,0.1);">
            <p style="margin:0; color:#a0a0a0;">Characters</p>
            <h3 style="margin:0;">{l}</h3>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="margin:0; color:#a0a0a0;">Words</p>
            <h3 style="margin:0;">{w}</h3>
            <hr style="border-color: rgba(255,255,255,0.1);">
            <p style="margin:0; color:#a0a0a0;">Data Status</p>
            <h3 style="color: #00d2ff; margin:0;">Valid</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting Input Stream...")
        st.markdown("Use the panel on the left to input problem data.")

# --- 6. PREDICTION LOGIC ---
if analyze_btn:
    if not desc:
        st.warning("Input Buffer Empty. Please provide a description.")
    else:
        # UX: Loading Animation
        progress_text = "Processing neural vectors..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)

        # Logic
        combined = f"{title} {desc} {in_desc} {out_desc}"
        feats = get_manual_features(combined)

        clean = clean_text(combined)
        tfidf_vec = tfidf_vectorizer.transform([clean])
        manual_features = np.array(feats).reshape(1, -1)
        final_input = hstack((tfidf_vec, manual_features))

        pred_class = clf_model.predict(final_input)[0]
        pred_score = reg_model.predict(final_input)[0]

        my_bar.empty()

        # --- RESULTS SECTION ---
        st.markdown("---")
        st.subheader("Analysis Report")

        # Custom Result Cards
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)

        with res_c1:
            st.markdown("**Difficulty Class**")

            # --- FIX: ROBUST LOWERCASE CHECK ---
            p_class = str(pred_class).lower().strip()

            if p_class == "easy":
                bg = "#00c853"  # Green
                label_color = "#ffffff"
            elif p_class == "medium":
                bg = "#ffab00"  # Amber/Orange
                label_color = "#000000"
            else:
                bg = "#d50000"  # Red
                label_color = "#ffffff"

            st.markdown(f"""
            <div style="background:{bg}; padding:10px; border-radius:8px; text-align:center;">
                <h3 style="margin:0; color:{label_color}; text-shadow: 0px 1px 2px rgba(0,0,0,0.3);">{pred_class.upper()}</h3>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.metric("Predicted Rating", int(pred_score), "+ AI Est.")

        with res_c3:
            st.metric("Math Density", feats[1], "Symbols")

        with res_c4:
            st.metric("Algo Keywords", feats[2], "Detected")

        # Visualization Row
        st.markdown("")
        st.markdown("#### Feature Influence Vector")

        chart_data = pd.DataFrame({
            "Feature": ["Text Length", "Math Symbols", "Keywords", "Numeric Constants"],
            "Intensity": [feats[0] / 100, feats[1], feats[2], feats[5]]
        })

        st.bar_chart(chart_data, x="Feature", y="Intensity", color="#3a7bd5")

        # Developer Dump
        with st.expander("Developer Logs (JSON Dump)"):
            st.json({
                "timestamp": time.time(),
                "model_version": "v2.4",
                "prediction": {
                    "class": pred_class,
                    "score": pred_score
                },
                "features": feats
            })
