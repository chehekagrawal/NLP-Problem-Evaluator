from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from src.utils import log_message


def train_regressors(X_train, y_train, X_test, y_test):
    log_message("\nPHASE 3: REGRESSION BATTLE")

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    results = {}
    best_model = None
    best_mae = float('inf')

    for name, model in regressors.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = mae
        log_message(f"Regressor: {name} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model

    log_message(f"\nBest Regressor: {best_model.__class__.__name__} with MAE: {best_mae:.2f}")
    return best_model, results