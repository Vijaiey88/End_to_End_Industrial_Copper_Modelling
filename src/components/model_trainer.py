import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor  # Import BaggingRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Bagging Regressor": BaggingRegressor(n_estimators=150, random_state=6)
            }
            params = {
                "Decision Tree": {
                    'criterion': ['mse', 'friedman_mse', 'mae'],
                },
                "Bagging Regressor": {}  # You can add BaggingRegressor hyperparameters here if needed
            }
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best model's performance is below the threshold.")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            # Calculate performance metrics for Bagging Regressor
            performance_metrics = {
                "R2 Score": r2_square,
                "Mean Squared Error": mean_squared_error(y_test, predicted),
                "Mean Absolute Error": mean_absolute_error(y_test, predicted),
                "Root Mean Squared Error": np.sqrt(mean_squared_error(y_test, predicted))
            }

            return performance_metrics

        except Exception as e:
            raise CustomException(e, sys)
