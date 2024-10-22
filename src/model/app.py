import streamlit as st
import pandas as pd
import mlflow
import dagshub
import logging

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load the dataset
df = pd.read_csv('data/processed/train.csv')

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Kushagra-Bisht', repo_name='Smartphone_price_prediction', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Smartphone_price_prediction.mlflow')

model_name = "model" 
model_version = 2
model_uri = f'models:/{model_name}/{model_version}'

# Cache the model loading
@st.cache_resource()
def load_model():
    return mlflow.pyfunc.load_model(model_uri)

# Load the model
model = load_model()

# Streamlit interface
st.title("Smartphone Price Prediction")
st.write("Enter the features to predict the price of a smartphone:")

# Input features
feature1 = st.number_input("Rating", min_value=0.0, max_value=5.0)
feature2 = st.number_input("Battery", min_value=df['battery'].min(), max_value=df['battery'].max())
feature3 = st.selectbox("Brand", options=df['brand'].value_counts().index.tolist())
feature4 = st.number_input("RAM", min_value=df['ram'].min(), max_value=df['ram'].max())
feature5 = st.number_input("ROM", min_value=df['rom'].min(), max_value=df['rom'].max())
feature6 = st.number_input("Front Camera", min_value=df['front_camera'].min(), max_value=df['front_camera'].max())
feature7 = st.selectbox("Processor", options=df['processor_company'].value_counts().index.tolist())
feature8 = st.number_input("Display Size", min_value=df['display_size'].min(), max_value=df['display_size'].max())
feature9 = st.number_input("Primary Rear Camera", min_value=df['primary_rear_camera'].min(), max_value=df['primary_rear_camera'].max())
feature10 = st.number_input("No. of Cameras", min_value=df['no.of cameras'].min(), max_value=df['no.of cameras'].max())

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'rating': feature1,
        'battery': feature2,
        'brand': feature3,
        'ram': feature4,
        'rom': feature5,
        'front_camera': feature6,
        'processor_company': feature7,
        'display_size': feature8,
        'primary_rear_camera': feature9,
        'no.of cameras': feature10
    }])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f"Prediction: {prediction[0]}")

