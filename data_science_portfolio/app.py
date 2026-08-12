import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction App")
st.write("An interactive machine learning app built to predict passenger survival based on real historical data.")

# Load Data Caching for performance
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/titanic_clean.csv")
    return df

df = load_data()

# Train Model on the fly
@st.cache_resource
def train_model():
    X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']]
    X = pd.get_dummies(X, drop_first=True)
    y = df['survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

model, feature_columns = train_model()

# Sidebar User Inputs
st.sidebar.header("Passenger Parameters")

pclass = st.sidebar.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 28)
fare = st.sidebar.slider("Ticket Fare (£)", 0.0, 500.0, 32.0)
sibsp = st.sidebar.slider("Siblings / Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents / Children Aboard", 0, 6, 0)

# Main Panel Display
col1, col2 = st.columns(2)
with col1:
    st.subheader("Quick Dataset Preview")
    st.dataframe(df.head(5))

with col2:
    st.subheader("Your Input Profile")
    input_data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare]
    })
    st.write(input_data)

# Prediction Section
st.divider()
if st.button("Predict Survival", type="primary"):
    # Process inputs to match training columns
    processed_input = pd.get_dummies(input_data, drop_first=True)
    processed_input = processed_input.reindex(columns=feature_columns, fill_value=0)
    
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0]
    
    if prediction == 1:
        st.success(f"🎉 **Result: Survived!** (Confidence: {prediction_proba[1]*100:.1f}%)")
    else:
        st.error(f"⚠️ **Result: Did not survive.** (Confidence: {prediction_proba[0]*100:.1f}%)")