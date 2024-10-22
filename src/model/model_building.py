import pickle
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import pickle
import yaml
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameter retreived successfully")
        return params
    except Exception as e:
        logger.error('Unexpected error:',e)
        raise



class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.missing_cols = []  # Initialize a list to store names of missing columns

    def fit(self, X, y=None):
        # Store the names of columns with missing values
        self.missing_cols = [col for col in X.columns if X[col].isna().sum() > 0]
        return self

    def transform(self, X):
        # Create a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()
        
        for col in self.missing_cols:
            if X_transformed[col].dtype == 'object':
                # For categorical columns, fill missing values with the mode
                mode_value = X_transformed[col].mode()[0]  # Get the mode
                X_transformed[col].fillna(mode_value, inplace=True)
            else:
                # For numeric columns, check for outliers and fill accordingly
                if self.check_outliers(X_transformed, col):
                    X_transformed[col] = X_transformed.groupby('brand')[col].transform(lambda x: x.fillna(x.median()))
                else:
                    X_transformed[col] = X_transformed.groupby('brand')[col].transform(lambda x: x.fillna(x.mean()))
        
        return X_transformed

    def check_outliers(self, df, col):
        # Calculate median and IQR
        median = df[col].median()
        q3 = df[col].quantile(0.75)
        q1 = df[col].quantile(0.25)
        iqr = q3 - q1

        # Determine bounds for outliers
        inliers_start = median - 1.5 * iqr
        inliers_end = median + 1.5 * iqr

        # Calculate the proportion of outliers
        outliers_count = df[(df[col] > inliers_end) | (df[col] < inliers_start)].shape[0]
        outlier_proportion = outliers_count / df.shape[0]

        return outlier_proportion >= 0.1 


numeric_features = ['battery', 'ram', 'rom', 'front_camera', 'display_size', 'primary_rear_camera','no.of cameras','rating']
categorical_features = ['brand', 'processor_company']
preprocessor = ColumnTransformer(
transformers=[('num', StandardScaler(), numeric_features),('target_encoder', ce.TargetEncoder(), categorical_features)]
        )
# Set output of the preprocessor to return a pandas DataFrame
preprocessor.set_output(transform='pandas')

def train_model(X_train, y_train):
    try:
        numeric_features = ['battery', 'ram', 'rom', 'front_camera', 'display_size', 'primary_rear_camera','no.of cameras','rating']
        categorical_features = ['brand', 'processor_company']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('target_encoder', ce.TargetEncoder(), categorical_features)
            ]
        )
        # Set output of the preprocessor to return a pandas DataFrame
        preprocessor.set_output(transform='pandas')

        # Pipeline
        pipeline = Pipeline(steps=[
            ('imputer', MissingValueImputer()),
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                learning_rate=0.15933315661811834,
                n_estimators=272,
                max_depth=9,
                min_child_weight=1,
                subsample=0.8261207403464462,
                colsample_bytree=0.6221149729127035,
                gamma=4.48032662619819,
                reg_alpha=0.8688526217318796,
                reg_lambda=0.4660114566903573,
                random_state=42
            ))
        ])

        # Fit the model
        pipeline.fit(X_train, y_train)
        return pipeline

    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise


def save_model(model, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data('data/processed/train.csv')
        numeric_features = ['battery', 'ram', 'rom', 'front_camera', 'display_size', 'primary_rear_camera','no.of cameras','rating']
        categorical_features = ['brand', 'processor_company']
        X_train = train_data.drop(columns=['price'])
        y_train = train_data['price']
        pipeline = train_model(X_train, y_train)
        save_model(pipeline, 'models/model.pkl')
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
