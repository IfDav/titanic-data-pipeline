import os
import pandas as pd
import seaborn as sns

def extract_data() -> pd.DataFrame:
    """Step 1: Extract raw data from the source."""
    print("Extracting raw data...")
    df = sns.load_dataset('titanic')
    return df

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Transform and clean the data."""
    print("Transforming and cleaning data...")
    
    # Drop unnecessary columns
    df_clean = df.drop(columns=['deck', 'embark_town', 'alive'], errors='ignore')
    
    # Fill missing ages with the median age
    median_age = df_clean['age'].median()
    df_clean['age'] = df_clean['age'].fillna(median_age)
    
    # Drop any remaining missing rows
    df_clean = df_clean.dropna()
    
    print(f"Data cleaned. Shape went from {df.shape} to {df_clean.shape}")
    return df_clean

def load_data(df: pd.DataFrame, output_path: str):
    """Step 3: Load/Save the cleaned data into the processed folder."""
    print(f"Loading/Saving clean data to {output_path}...")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as a clean CSV file
    df.to_csv(output_path, index=False)
    print("ETL Pipeline executed successfully!")

if __name__ == "__main__":
    raw_data = extract_data()
    clean_data = transform_data(raw_data)
    load_data(clean_data, "data/processed/titanic_clean.csv")