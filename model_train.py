import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

    def train(self, X, y):
        self.weights = np.zeros(self.n)
        self.bias = 0

        for it in range(self.iters):
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            dz = y_predict - y
            dw = (1 / self.m) * np.dot(X.T, dz)
            db = (1 / self.m) * np.sum(dz)

            self.weights -= (dw * self.lr)
            self.bias -= (db * self.lr)

    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        y_predict = self.predict_proba(X)
        return np.array(['Yes' if prob >= self.threshold else 'No' for prob in y_predict])

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(URL)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

X = df.drop(["Churn", "customerID"], axis=1)
y_encoded = np.array([1 if val == 'Yes' else 0 for val in df["Churn"]])

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

processor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
])

X_processed = processor.fit_transform(X)

my_model = LogisticRegressionCustom(X=X_processed, no_iterations=2000, learningrate=0.1, threshold=0.3)
my_model.train(X_processed, y_encoded)

joblib.dump({'processor': processor, 'model': my_model}, 'churn_model.pkl')
print("Model and processor saved successfully!")