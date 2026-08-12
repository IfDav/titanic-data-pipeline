import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_titanic_model(data_path: str):
    print("Loading clean data...")
    df = pd.read_csv(data_path)
    
    # Define features (X) and target (y)
    # We'll use pclass, sex, age, sibsp, parch, and fare as basic features
    # Note: We need to convert text columns like 'sex' into numbers using pandas get_dummies
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
    X = pd.get_dummies(X, drop_first=True) # Converts 'sex' (male/female) into a 0 or 1
    
    y = df['survived']
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nModel Training Complete!")
    print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    train_titanic_model("data/processed/titanic_clean.csv")