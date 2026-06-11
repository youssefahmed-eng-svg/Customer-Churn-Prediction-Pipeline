import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(URL)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

x = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

processing = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
])

model_pipeline = Pipeline([
    ("processor", processing),
    ("estimator", RandomForestClassifier(random_state=42))
])

param_grid = {
    "estimator__n_estimators": [50, 100, 150],
    "estimator__max_depth": [2, 5, 7]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="recall_macro")
grid_search.fit(x, y)

print(f"Best Score: {grid_search.best_score_}")
print(f"Best Params: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x)

print("\n--- Classification Report ---")
print(classification_report(y, y_pred))
