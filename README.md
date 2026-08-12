# Titanic Survival Prediction Pipeline & Web App

An end-to-end data science portfolio project featuring an automated ETL pipeline, a Random Forest machine learning classifier, and an interactive web application built with Streamlit to predict passenger survival on the Titanic.

## Project Structure
- `src/etl.py`: Extracts raw Seaborn data, cleans missing values, handles data types, and outputs a processed dataset.
- `src/train.py`: Loads clean data, engineers features using pandas, and trains a Random Forest model.
- `app.py`: An interactive Streamlit web dashboard for real-time model predictions and parameter tuning.

## Results
- **Model Accuracy:** 77%
- **Algorithm:** Random Forest Classifier (Scikit-Learn)

## How to Run Locally
1. Clone the repository and set up a virtual environment.
2. Install dependencies: 
   ```bash
   pip install pandas scikit-learn streamlit
