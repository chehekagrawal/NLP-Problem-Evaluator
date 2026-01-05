import pandas as pd
import joblib
import os
import json
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Import our new modules
from src.utils import setup_directories, setup_logging, log_message, MODELS_DIR
from src.eda import load_and_analyze_data
from src.preprocessing import clean_text
from src.features import get_manual_features
from src.classification import train_classifiers
from src.regression import train_regressors
from src.plotting import (plot_feature_correlation, plot_correlation_heatmap,
                          plot_model_comparison, plot_confusion_matrix, plot_prediction_scatter)


def train_pipeline():
    # 1. Setup
    setup_directories()
    setup_logging()

    # 2. EDA & Loading
    df = load_and_analyze_data()
    if df is None: return

    # 3. Feature Engineering
    log_message("\nPHASE 2: FEATURE ENGINEERING")

    df['combined_text'] = (
            df['title'] + " " + df['description'] + " " +
            df['input_description'] + " " + df['output_description']
    )

    feature_data = df['combined_text'].apply(get_manual_features).tolist()
    feature_cols = ['text_len', 'math_symbols', 'keyword_freq', 'word_count', 'avg_word_len', 'number_count']
    df_features = pd.DataFrame(feature_data, columns=feature_cols)
    df = pd.concat([df, df_features], axis=1)

    # Plot Features
    plot_feature_correlation(df, 'math_symbols')
    plot_correlation_heatmap(df, feature_cols + ['problem_score'])

    log_message(f"Engineered Features: {', '.join(feature_cols)}")

    # 4. Preprocessing & Vectorization
    log_message("Vectorizing (TF-IDF)...")
    df['clean_text'] = df['combined_text'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=3000)
    tfidf_matrix = tfidf.fit_transform(df['clean_text'])

    # 5. Split Data
    X_manual = df[feature_cols].values
    X = hstack((tfidf_matrix, X_manual))
    y_class = df['problem_class']
    y_score = df['problem_score']

    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42
    )

    # 6. Train Models
    best_clf, clf_results = train_classifiers(X_train, y_class_train, X_test, y_class_test)
    best_reg, reg_results = train_regressors(X_train, y_score_train, X_test, y_score_test)

    # 7. Final Plots
    plot_model_comparison(clf_results, '3_model_comparison_class.png', 'Accuracy', 'cornflowerblue')
    plot_model_comparison(reg_results, '5_model_comparison_reg.png', 'MAE (Lower is Better)', 'salmon')

    plot_confusion_matrix(y_class_test, best_clf.predict(X_test), best_clf.classes_, best_clf.__class__.__name__)
    plot_prediction_scatter(y_score_test, best_reg.predict(X_test), best_reg.__class__.__name__)

    # 8. Save Models
    log_message("\nPHASE 4: SAVING MODELS")
    joblib.dump(best_clf, os.path.join(MODELS_DIR, 'classifier_model.pkl'))
    joblib.dump(best_reg, os.path.join(MODELS_DIR, 'regressor_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

    # Save Metadata
    feature_metadata = {
        'manual_features': feature_cols,
        'tfidf_features': tfidf.get_feature_names_out().tolist()[:50]
    }
    with open(os.path.join(MODELS_DIR, 'feature_metadata.json'), 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    log_message("Pipeline Finished Successfully.")


if __name__ == "__main__":
    train_pipeline()