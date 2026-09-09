import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

class LogisticRegressionCustom:
    def __init__(self, X, no_iterations, learningrate, threshold=0.3):
        self.lr = learningrate
        self.iters = no_iterations
        self.threshold = threshold
        self.m, self.n = X.shape if X is not None else (0, 0)
        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        y_predict = self.predict_proba(X)
        return np.array(['Yes' if prob >= self.threshold else 'No' for prob in y_predict])

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'churn_model.pkl')

saved_data = joblib.load(model_path)
processor = saved_data['processor']
model = saved_data['model']

st.title("📊 Customer Churn Prediction System")

mode = st.radio("Choose Prediction Mode:", ["Batch Prediction (Upload CSV)", "Single Customer Input"])

if mode == "Batch Prediction (Upload CSV)":
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", df_input.head())
        
        if st.button("Run Predictions for All Customers"):
            processed_data = processor.transform(df_input)
            predictions = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)
            
            df_input['Churn Prediction'] = predictions
            df_input['Churn Risk (%)'] = np.round(probabilities * 100, 2)
            
            st.success("✅ Analysis Completed!")
            st.dataframe(df_input)
            
            csv_data = df_input.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction Results", data=csv_data, file_name="churn_predictions.csv", mime="text/csv")

else:
    tenure = st.number_input("Tenure", 0, 100, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    if st.button("Predict"):
        data = pd.DataFrame([{
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': monthly,
            'TotalCharges': total
        }])

        processed_data = processor.transform(data)
        pred = model.predict(processed_data)
        proba = model.predict_proba(processed_data)[0]

        st.write(f"Result: Churn (Risk: {proba*100:.1f}%)" if pred[0] == 'Yes' else f"Result: Stay (Risk: {proba*100:.1f}%)")