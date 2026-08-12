# Titanic Survival Prediction Pipeline

An end-to-end data science portfolio project featuring an automated ETL pipeline and a Random Forest machine learning classifier to predict passenger survival on the Titanic.

## Project Structure
- `src/etl.py`: Extracts raw Seaborn data, cleans missing values, handles data types, and outputs a processed dataset.
- `src/train.py`: Loads clean data, engineers features using pandas, and trains a Random Forest model.

## Results
- **Model Accuracy:** 77%
- **Algorithm:** Random Forest Classifier (Scikit-Learn)

## How to Run
1. Clone the repository and set up a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the ETL script: `python src/etl.py`
4. Run the training script: `python src/train.py`
