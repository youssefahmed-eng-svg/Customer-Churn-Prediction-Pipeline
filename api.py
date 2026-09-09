from fastapi import FastAPI
from pydantic import BaseModel
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

app = FastAPI(title="Customer Churn Prediction API")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str = 'Male'
    SeniorCitizen: int = 0
    Partner: str = 'No'
    Dependents: str = 'No'
    PhoneService: str = 'Yes'
    MultipleLines: str = 'No'
    InternetService: str = 'DSL'
    OnlineSecurity: str = 'No'
    OnlineBackup: str = 'No'
    DeviceProtection: str = 'No'
    TechSupport: str = 'No'
    StreamingTV: str = 'No'
    StreamingMovies: str = 'No'
    Contract: str = 'Month-to-month'
    PaperlessBilling: str = 'Yes'
    PaymentMethod: str = 'Electronic check'

@app.post("/predict")
def predict_churn(customer: CustomerData):
    data_df = pd.DataFrame([customer.dict()])
    processed_data = processor.transform(data_df)
    
    prediction = model.predict(processed_data)[0]
    probability = float(model.predict_proba(processed_data)[0])
    
    return {
        "churn_prediction": prediction,
        "churn_risk_percentage": round(probability * 100, 2)
    }