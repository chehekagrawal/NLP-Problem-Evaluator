import pandas as pd
import json
import os
import numpy as np
from src.utils import DATA_PATH, log_message
from src.plotting import plot_class_distribution, plot_score_distribution

def load_and_analyze_data():
    log_message("PHASE 1: DATA INGESTION & EDA")
    log_message(f"Loading dataset from {DATA_PATH}...")

    if not os.path.exists(DATA_PATH):
        log_message(f"Error: File not found at {DATA_PATH}")
        return None

    data_list = []
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        df = pd.DataFrame(data_list)
        log_message(f"Data loaded. Rows: {len(df)}")
    except Exception as e:
        log_message(f"Critical Error: {e}")
        return None

    # Missing Value Analysis
    log_message("\nMissing Values per Column:")
    log_message(str(df.isnull().sum()))

    # Imputation
    text_cols = ['title', 'description', 'input_description', 'output_description', 'problem_class']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if 'problem_score' in df.columns:
        df['problem_score'] = pd.to_numeric(df['problem_score'], errors='coerce')
        median_score = df['problem_score'].median()
        df['problem_score'] = df['problem_score'].fillna(median_score)
        log_message(f"Filled missing scores with Median: {median_score}")

    # Stats
    log_message(f"\nProblem Class Distribution:")
    log_message(str(df['problem_class'].value_counts()))

    if 'problem_score' in df.columns:
        log_message(f"\nProblem Score Statistics:")
        log_message(f"Mean: {df['problem_score'].mean():.2f}")
        log_message(f"Median: {df['problem_score'].median():.2f}")

    # Generate Plots
    plot_class_distribution(df)
    plot_score_distribution(df)

    return df