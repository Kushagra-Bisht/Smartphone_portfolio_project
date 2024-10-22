import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_rows', None)  # None means no limit on rows
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")
import logging
import yaml
import logging
import os
from sklearn.model_selection import train_test_split

# Create a logger
logger = logging.getLogger("data_transformation")
logger.setLevel(logging.DEBUG)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameter retreived successfully")
        return params
    except Exception as e:
        logger.error('Unexpected error:',e)
        raise

def load_data(data_path):
    try:
        df=pd.read_csv(data_path)
        logger.debug("Data loaded successfully")
        return df
    except Exception as e:
        logger.error("Failed to load data due to:",e)
        raise

def save_data(train_data,test_data,data_path):
    try:
        train_data.to_csv(os.path.join(data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(data_path,'test.csv'),index=False)
        logger.debug("Train and test data saved successfully")
    except Exception as e:
        logger.error('Unexpected error:',e)
        raise

def processor_cleaning(df):
    df = df.rename(columns={'processor2': 'processor_company'})
    df['processor_company']=df['processor_company'].str.lower()
    df['processor_company']=df['processor_company'].str.replace("qualcomm®","qualcomm")
    df['processor_company']=df['processor_company'].str.replace("meditek","mediatek")
    df['processor_company']=df['processor_company'].str.replace("snapdragon®","snapdragon")
    df['processor_company']=df['processor_company'].str.replace("dimesity","dimensity")
    df['processor_company']=df['processor_company'].str.replace("qualcomm?","qualcomm")
    df['processor_company']=df['processor_company'].str.replace("google","tensor")
    df.dropna(subset=['price'], inplace=True)
    return df

def main():
    try:
        df=pd.read_csv("data/interim/smartphones_final_clean.csv")
        logger.debug('data loaded properly')
        df=processor_cleaning(df)
        logger.debug('data preprocessed properly')
        params=load_params('params.yaml')
        test_size=params['data_ingestion']['test_size']
        train_data,test_data=train_test_split(df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='data/processed')  
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main() 