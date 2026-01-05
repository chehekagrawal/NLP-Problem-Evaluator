import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.utils import REPORTS_DIR, log_message


def plot_class_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='problem_class', data=df, color='steelblue')
    plt.title('Distribution of Problem Difficulty Classes')
    plt.ylabel('Count')
    plt.xlabel('Difficulty Class')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, '1_class_distribution.png')
    plt.savefig(save_path)
    plt.close()
    log_message(f"Saved plot: {save_path}")


def plot_score_distribution(df):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['problem_score'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Problem Scores')

    plt.subplot(1, 2, 2)
    df.boxplot(column='problem_score', by='problem_class', figsize=(6, 5))
    plt.suptitle('')
    plt.title('Score Distribution by Difficulty')
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, '1b_score_distribution.png')
    plt.savefig(save_path)
    plt.close()
    log_message(f"Saved plot: {save_path}")


def plot_feature_correlation(df, feature_name='math_symbols'):
    plt.figure(figsize=(10, 6))
    df_filtered = df[df['problem_class'].isin(df['problem_class'].unique())]
    sns.boxplot(x='problem_class', y=feature_name, data=df_filtered, color='lightblue')
    plt.title(f'{feature_name} Density by Difficulty')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, '2_feature_correlation.png')
    plt.savefig(save_path)
    plt.close()
    log_message(f"Saved plot: {save_path}")


def plot_correlation_heatmap(df, cols):
    plt.figure(figsize=(10, 8))
    numeric_df = df[cols].copy()
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, '2b_correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    log_message(f"Saved plot: {save_path}")


def plot_model_comparison(results, filename, title, color):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), color=color)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, classes, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '4_confusion_matrix.png'))
    plt.close()


def plot_prediction_scatter(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Actual vs Predicted ({model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '5b_prediction_scatter.png'))
    plt.close()