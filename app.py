import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack

# --- CONFIGURATION ---
PAGE_TITLE = "AutoJudge: AI Problem Difficulty Predictor"
PAGE_ICON = "⚡"

# --- SETUP PAGE ---
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")


# --- LOAD MODELS & ASSETS ---
@st.cache_resource
def load_assets():
    try:
        clf_model = joblib.load('models/classifier_model.pkl')
        reg_model = joblib.load('models/regressor_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        return clf_model, reg_model, tfidf
    except FileNotFoundError:
        st.error("Error: Model files not found. Please run 'train_model.py' first.")
        st.stop()


clf_model, reg_model, tfidf_vectorizer = load_assets()


# --- HELPER FUNCTIONS (Must match train_model.py exactly) ---
def get_features(text):
    # 1. Math Symbols
    math_symbols = ['+', '-', '*', '/', '=', '<', '>', '^', '_', '{', '}', '$', '%']
    math_count = sum(text.count(s) for s in math_symbols)

    # 2. Keywords
    keywords = ['graph', 'tree', 'dp', 'recursion', 'array', 'greedy', 'binary', 'modulo']
    keyword_freq = sum(text.lower().count(k) for k in keywords)

    # 3. Text Length
    text_len = len(text)

    # 4. Word Count
    words = text.split()
    word_count = len(words)

    # 5. Average Word Length
    if word_count > 0:
        avg_word_len = np.mean([len(w) for w in words])
    else:
        avg_word_len = 0

    # 6. Number Count
    number_count = len(re.findall(r'\d+', text))

    return [text_len, math_count, keyword_freq, word_count, avg_word_len, number_count]


def clean_text(text):
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])


# --- UI LAYOUT ---
st.title(PAGE_TITLE)
st.markdown("""
    This system uses Machine Learning to predict the difficulty of a programming problem 
    based on its textual description.

    **Instructions:** Paste the problem details below and click 'Predict'.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    problem_title = st.text_input("Problem Title (Optional)")

with col2:
    pass

problem_desc = st.text_area("Problem Description", height=200,
                            placeholder="Paste the main story of the problem here...")
input_desc = st.text_area("Input Description", height=100, placeholder="e.g., The first line contains an integer T...")
output_desc = st.text_area("Output Description", height=100, placeholder="e.g., Print the maximum sum...")

# --- PREDICTION LOGIC ---
if st.button("Predict Difficulty", type="primary"):
    if not problem_desc:
        st.warning("Please enter at least the Problem Description.")
    else:
        with st.spinner("Analyzing text complexity..."):
            # 1. Aggregate Text
            combined_text = f"{problem_title} {problem_desc} {input_desc} {output_desc}"

            # 2. Extract Manual Features
            # Order: text_len, math_symbols, keyword_freq, word_count, avg_word_len, number_count
            manual_features = np.array(get_features(combined_text)).reshape(1, -1)

            # 3. NLP & TF-IDF
            cleaned_text = clean_text(combined_text)
            tfidf_vector = tfidf_vectorizer.transform([cleaned_text])

            # 4. Stack Features
            # Note: Tfidf comes first, then manual features (matching train_model.py)
            final_input = hstack((tfidf_vector, manual_features))

            # 5. Predict
            predicted_class = clf_model.predict(final_input)[0]
            predicted_score = reg_model.predict(final_input)[0]

            # --- DISPLAY RESULTS ---
            st.success("Prediction Complete!")

            # Create metrics
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("Difficulty Class")
                # Color logic for display
                if predicted_class == "Easy":
                    color = "green"
                elif predicted_class == "Medium":
                    color = "orange"
                else:
                    color = "red"
                st.markdown(f":{color}[**{predicted_class}**]")

            with res_col2:
                st.subheader("Predicted Score")
                st.metric(label="Rating", value=f"{int(predicted_score)}")

            # Explanation / Debug Info (Professional Touch)
            with st.expander("View Analysis Details"):
                st.write("Feature Analysis:")
                st.json({
                    "Math Symbols Found": manual_features[0][1],
                    "Algorithmic Keywords": manual_features[0][2],
                    "Text Length (Chars)": manual_features[0][0],
                    "Word Count": manual_features[0][3]
                })