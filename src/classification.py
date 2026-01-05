from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from src.utils import log_message


def train_classifiers(X_train, y_train, X_test, y_test):
    log_message("\nPHASE 3: CLASSIFICATION BATTLE")

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (Linear)": LinearSVC(random_state=42, dual='auto')
    }

    results = {}
    best_model = None
    best_acc = 0

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        log_message(f"Classifier: {name} | Accuracy: {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model = model

    log_message(f"\nBest Classifier: {best_model.__class__.__name__} with Accuracy: {best_acc * 100:.2f}%")
    return best_model, results