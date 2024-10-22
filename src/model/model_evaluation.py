# updated model evaluation
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from model_building import MissingValueImputer

dagshub_token = "9eabf8ed2aa21c5b0eddab07ae318ba27e66ed6c"
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Smartphone_portfolio_project.mlflow')

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(pipeline, X_test, y_test):
    try:
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)  # Correct order: true values first
        r2 = r2_score(y_test, y_pred)  # Correct order: true values first

        metrics_dict = {
            'mean_absolute_error': mae,
            'r2_score': r2
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def main():

    experiment_name = "dvc-pipeline"
    
    # Check if the experiment exists
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create the experiment if it does not exist
            mlflow.create_experiment(experiment_name)
            logger.info(f"Experiment '{experiment_name}' created successfully.")
        else:
            logger.info(f"Experiment '{experiment_name}' already exists.")
    except Exception as e:
        logger.error(f"Error accessing or creating experiment: {e}")
        return  # Exit the function if unable to create or access the experiment

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        try:
            pipeline = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test.csv')

            X_test = test_data.drop(columns=['price'])
            y_test = test_data['price']

            # Evaluate model and get metrics
            metrics = evaluate_model(pipeline, X_test, y_test)

            save_metrics(metrics, 'reports/metrics.json')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(pipeline, 'get_params'):
                params = pipeline.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Infer the signature
            signature = infer_signature(X_test, pipeline.predict(X_test))

            # Log model with signature
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)

            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

            logger.info("Model evaluation completed successfully.")

        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
