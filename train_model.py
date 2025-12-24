import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVC

# Metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix

# --- CONFIGURATION ---
DATA_PATH = 'data/problem_data.jsonl'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'

# 1. GLOBAL WARNING SUPPRESSION
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Logging Setup
log_file_path = os.path.join(REPORTS_DIR, 'experiment_log.txt')
def log_message(message):
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write(message + "\n")

# Clear previous log
with open(log_file_path, 'w', encoding='utf-8') as f:
    f.write("--- AutoJudge Experiment Log ---\n\n")

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==========================================
# PHASE 1: DATA INGESTION & EDA
# ==========================================
log_message("PHASE 1: DATA INGESTION & EDA")
log_message(f"Loading dataset from {DATA_PATH}...")

if not os.path.exists(DATA_PATH):
    log_message(f"Error: File not found at {DATA_PATH}")
    exit()

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
    exit()

# 1.1 Missing Value Analysis
missing_summary = df.isnull().sum()
log_message("\nMissing Values per Column:")
log_message(str(missing_summary))

# 1.2 Imputation
text_cols = ['title', 'description', 'input_description', 'output_description', 'problem_class']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

if 'problem_score' in df.columns:
    df['problem_score'] = pd.to_numeric(df['problem_score'], errors='coerce')
    median_score = df['problem_score'].median()
    df['problem_score'] = df['problem_score'].fillna(median_score)
    log_message(f"Filled missing scores with Median: {median_score}")

# 1.3 Data Quality Check
log_message(f"\nDataset Shape: {df.shape}")
log_message(f"Columns: {list(df.columns)}")
log_message(f"\nProblem Class Distribution:")
class_counts = df['problem_class'].value_counts()
log_message(str(class_counts))

# 1.4 Score Statistics
if 'problem_score' in df.columns:
    log_message(f"\nProblem Score Statistics:")
    log_message(f"Mean: {df['problem_score'].mean():.2f}")
    log_message(f"Median: {df['problem_score'].median():.2f}")
    log_message(f"Std Dev: {df['problem_score'].std():.2f}")
    log_message(f"Min: {df['problem_score'].min():.2f}")
    log_message(f"Max: {df['problem_score'].max():.2f}")

# 1.5 Visual Proof: Class Distribution (FIXED)
plt.figure(figsize=(8, 5))
# Get actual classes present in data
actual_classes = df['problem_class'].value_counts().sort_index()
sns.countplot(x='problem_class', data=df, color='steelblue')
plt.title('Distribution of Problem Difficulty Classes')
plt.ylabel('Count')
plt.xlabel('Difficulty Class')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '1_class_distribution.png'))
plt.close()
log_message("Saved plot: reports/1_class_distribution.png")

# 1.6 Visual Proof: Score Distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['problem_score'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Problem Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.subplot(1, 2, 2)
df.boxplot(column='problem_score', by='problem_class', figsize=(6, 5))
plt.suptitle('')
plt.title('Score Distribution by Difficulty Class')
plt.xlabel('Difficulty Class')
plt.ylabel('Problem Score')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '1b_score_distribution.png'))
plt.close()
log_message("Saved plot: reports/1b_score_distribution.png")

# ==========================================
# PHASE 2: FEATURE ENGINEERING
# ==========================================
log_message("\nPHASE 2: FEATURE ENGINEERING")

df['combined_text'] = (
    df['title'] + " " + df['description'] + " " +
    df['input_description'] + " " + df['output_description']
)

# Feature A: Text Length
df['text_len'] = df['combined_text'].apply(len)

# Feature B: Math Symbols
math_symbols = ['+', '-', '*', '/', '=', '<', '>', '^', '_', '{', '}', '$', '%']
def count_math(text):
    return sum(text.count(s) for s in math_symbols)
df['math_symbols'] = df['combined_text'].apply(count_math)

# Feature C: Keywords
keywords = ['graph', 'tree', 'dp', 'recursion', 'array', 'greedy', 'binary', 'modulo']
def count_keys(text):
    return sum(text.lower().count(k) for k in keywords)
df['keyword_freq'] = df['combined_text'].apply(count_keys)

# Feature D: Word Count
df['word_count'] = df['combined_text'].apply(lambda x: len(x.split()))

# Feature E: Average Word Length
df['avg_word_len'] = df['combined_text'].apply(
    lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
)

# Feature F: Number Count
df['number_count'] = df['combined_text'].apply(lambda x: len(re.findall(r'\d+', x)))

log_message(f"Engineered Features: text_len, math_symbols, keyword_freq, word_count, avg_word_len, number_count")

# 2.1 Feature Statistics by Class
log_message("\nFeature Statistics by Difficulty Class:")
feature_cols = ['text_len', 'math_symbols', 'keyword_freq', 'word_count', 'avg_word_len', 'number_count']
for feat in feature_cols:
    log_message(f"\n{feat}:")
    log_message(str(df.groupby('problem_class')[feat].describe()))

# 2.2 Visual Proof: Feature vs Difficulty (FIXED)
plt.figure(figsize=(10, 6))
# Only plot classes that exist in the data
df_filtered = df[df['problem_class'].isin(df['problem_class'].unique())]
sns.boxplot(x='problem_class', y='math_symbols', data=df_filtered, color='lightblue')
plt.title('Math Symbol Density by Difficulty')
plt.xlabel('Difficulty Class')
plt.ylabel('Math Symbol Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '2_feature_correlation.png'))
plt.close()
log_message("Saved plot: reports/2_feature_correlation.png")

# 2.3 Visual Proof: Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_features = df[['text_len', 'math_symbols', 'keyword_freq', 'word_count',
                        'avg_word_len', 'number_count', 'problem_score']]
correlation_matrix = numeric_features.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '2b_correlation_heatmap.png'))
plt.close()
log_message("Saved plot: reports/2b_correlation_heatmap.png")

# 2.4 Visual Proof: Multi-Feature Comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features_to_plot = ['text_len', 'word_count', 'math_symbols', 'keyword_freq', 'avg_word_len', 'number_count']
for idx, feat in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    df_filtered.boxplot(column=feat, by='problem_class', ax=ax)
    ax.set_title(f'{feat} by Difficulty')
    ax.set_xlabel('Difficulty Class')
    ax.set_ylabel(feat)
    plt.sca(ax)
    plt.xticks(rotation=45)
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '2c_all_features_by_class.png'))
plt.close()
log_message("Saved plot: reports/2c_all_features_by_class.png")

# NLP Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return " ".join([w for w in text.split() if w not in stop_words])

df['clean_text'] = df['combined_text'].apply(clean_text)

# TF-IDF
log_message("Generating TF-IDF Vectors...")
tfidf = TfidfVectorizer(max_features=3000)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])

# Stack Features
X_manual = df[['text_len', 'math_symbols', 'keyword_freq', 'word_count', 'avg_word_len', 'number_count']].values
X = hstack((tfidf_matrix, X_manual))
y_class = df['problem_class']
y_score = df['problem_score']

# Split
X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X, y_class, y_score, test_size=0.2, random_state=42
)

log_message(f"\nTrain Set Size: {X_train.shape[0]}")
log_message(f"Test Set Size: {X_test.shape[0]}")
log_message(f"Feature Dimensions: {X_train.shape[1]}")

# ==========================================
# PHASE 3: MODEL TRAINING & SELECTION
# ==========================================
log_message("\nPHASE 3: MODEL BATTLE")

# --- Classification ---
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)": LinearSVC(random_state=42, dual='auto')
}

clf_results = {}
best_clf_model = None
best_acc = 0

for name, model in classifiers.items():
    model.fit(X_train, y_class_train)
    acc = accuracy_score(y_class_test, model.predict(X_test))
    clf_results[name] = acc
    log_message(f"Classifier: {name} | Accuracy: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_clf_model = model

# Visual Proof: Classifier Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(clf_results.keys()), y=list(clf_results.values()), color='cornflowerblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '3_model_comparison_class.png'))
plt.close()

# Visual Proof: Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_class_test, best_clf_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_clf_model.classes_,
            yticklabels=best_clf_model.classes_)
plt.title(f'Confusion Matrix ({best_clf_model.__class__.__name__})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '4_confusion_matrix.png'))
plt.close()

log_message(f"\nBest Classifier: {best_clf_model.__class__.__name__} with Accuracy: {best_acc*100:.2f}%")

# --- Regression ---
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

reg_results = {}
best_reg_model = None
best_mae = float('inf')

for name, model in regressors.items():
    model.fit(X_train, y_score_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_score_test, preds)
    rmse = np.sqrt(mean_squared_error(y_score_test, preds))
    reg_results[name] = mae
    log_message(f"Regressor: {name} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")
    if mae < best_mae:
        best_mae = mae
        best_reg_model = model

# Visual Proof: Regressor Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(reg_results.keys()), y=list(reg_results.values()), color='salmon')
plt.title('Regression Error (Lower is Better)')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '5_model_comparison_reg.png'))
plt.close()

# Visual Proof: Prediction vs Actual
plt.figure(figsize=(8, 6))
y_pred = best_reg_model.predict(X_test)
plt.scatter(y_score_test, y_pred, alpha=0.5, color='purple')
plt.plot([y_score_test.min(), y_score_test.max()],
         [y_score_test.min(), y_score_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title(f'Actual vs Predicted Scores ({best_reg_model.__class__.__name__})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, '5b_prediction_scatter.png'))
plt.close()

log_message(f"\nBest Regressor: {best_reg_model.__class__.__name__} with MAE: {best_mae:.2f}")

# ==========================================
# PHASE 4: SAVING ARTIFACTS
# ==========================================
log_message("\nPHASE 4: SAVING MODELS")
joblib.dump(best_clf_model, os.path.join(MODELS_DIR, 'classifier_model.pkl'))
joblib.dump(best_reg_model, os.path.join(MODELS_DIR, 'regressor_model.pkl'))
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

# Save feature names for later use
feature_metadata = {
    'manual_features': ['text_len', 'math_symbols', 'keyword_freq', 'word_count', 'avg_word_len', 'number_count'],
    'tfidf_features': tfidf.get_feature_names_out().tolist()[:50]  # Store top 50 for reference
}
with open(os.path.join(MODELS_DIR, 'feature_metadata.json'), 'w') as f:
    json.dump(feature_metadata, f, indent=2)

log_message("\nSaved Models:")
log_message("- classifier_model.pkl")
log_message("- regressor_model.pkl")
log_message("- tfidf_vectorizer.pkl")
log_message("- feature_metadata.json")

log_message("\nProcess Complete. Check 'reports/' folder for proofs.")
print("\nTask 3 Complete. Check the 'reports' folder!")
print(f"Generated {len([f for f in os.listdir(REPORTS_DIR) if f.endswith('.png')])} visualization plots.")