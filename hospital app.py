# ---------------------------------------------------
# Hospital Data Analysis & Prediction Dashboard
# (With Dataset Upload, EDA, Model Training, and Prediction)
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------
st.set_page_config(page_title="Hospital ML Dashboard", layout="wide")

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Dataset", "EDA", "Train Models", "Predict Disease"]
)

# ---------------------------------------------------
# Global Data Handler
# ---------------------------------------------------
DATA_PATH = "synthetic_hospital_patient_records.csv"

def save_uploaded_data(file):
    """Save uploaded CSV file locally and return DataFrame."""
    df = pd.read_csv(file)
    df.to_csv(DATA_PATH, index=False)
    return df

@st.cache_data
def load_data():
    """Load the dataset from the saved path."""
    return pd.read_csv(DATA_PATH)

# ---------------------------------------------------
# PAGE 1: Upload Dataset
# ---------------------------------------------------
if page == "Upload Dataset":
    st.title("Upload a New Hospital Dataset")
    st.write("Upload a CSV file to analyze, train, and predict from new data.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = save_uploaded_data(uploaded_file)
        st.success("File uploaded and saved successfully!")
        st.write("Preview of Uploaded Data")
        st.dataframe(df.head())

        st.write("Summary Info")
        st.write(df.describe())

        st.info("You can now move to EDA, Train Models, or Predict Disease tabs.")
    else:
        if os.path.exists(DATA_PATH):
            df = load_data()
            st.info("Using existing dataset:")
            st.dataframe(df.head())
        else:
            st.warning("No dataset found. Please upload a CSV file to proceed.")

# ---------------------------------------------------
# PAGE 2: EDA
# ---------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")

    if not os.path.exists(DATA_PATH):
        st.error("No dataset found. Please upload one in the 'Upload Dataset' tab.")
        st.stop()

    df = load_data()

    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.write("Missing Values")
    st.write(df.isnull().sum())

    # Disease distribution
    if "Disease" in df.columns:
        st.subheader("Disease Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Disease', palette='pastel', ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Numeric feature distribution
    st.subheader("Feature Distribution")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) > 0:
        selected_col = st.selectbox("Select a numeric column:", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, color='lightblue', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found!")

# ---------------------------------------------------
# PAGE 3: Train Models
# ---------------------------------------------------
elif page == "Train Models":
    st.title("Train Multiple Machine Learning Models")

    if not os.path.exists(DATA_PATH):
        st.error("No dataset found. Please upload one in the 'Upload Dataset' tab.")
        st.stop()

    df = load_data()
    df_encoded = df.copy()

    # Encode categorical features
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=['object']).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # Split data
    if "Disease" not in df_encoded.columns:
        st.error("'Disease' column not found — cannot train models.")
        st.stop()

    if "patient_ID" in df_encoded.columns:
        X = df_encoded.drop(columns=["Disease", "patient_ID"])
    else:
        X = df_encoded.drop(columns=["Disease"])

    y = df_encoded["Disease"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc})

    results_df = pd.DataFrame(results)
    st.write("Model Comparison")
    st.dataframe(results_df)

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=results_df, palette="mako", ax=ax)
    for i, val in enumerate(results_df["Accuracy"]):
        ax.text(i, val + 0.005, f"{val:.2f}", ha='center')
    st.pyplot(fig)

    best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, "best_hospital_model.pkl")
    st.success(f"Best Model: {best_model_name} (Saved as best_hospital_model.pkl)")

# ---------------------------------------------------
# PAGE 4: Predict Disease
# ---------------------------------------------------
elif page == "Predict Disease":
    st.title("Predict Disease from Patient Data")

    if not os.path.exists("best_hospital_model.pkl"):
        st.error("No model found! Please train one on the 'Train Models' tab first.")
        st.stop()

    df = load_data()
    model = joblib.load("best_hospital_model.pkl")

    st.subheader("Enter Patient Information")

    age = st.number_input("Age", min_value=0, max_value=100, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoker = st.selectbox("Smoker", ["Yes", "No"])

    # Encoding and scaling
    gender_encoded = 1 if gender == "Male" else 0
    smoker_encoded = 1 if smoker == "Yes" else 0

    input_data = np.array([[age, bmi, bp, chol, gender_encoded, smoker_encoded]])

    X = df.drop(columns=["Disease", "patient_ID"], errors='ignore')
    scaler = StandardScaler().fit(X)
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"High Risk of Disease (Probability: {prob:.2f})")
        else:
            st.success(f"Low Risk of Disease (Probability: {prob:.2f})")
