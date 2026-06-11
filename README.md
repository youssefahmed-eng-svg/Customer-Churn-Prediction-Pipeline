# Telco Customer Churn Prediction Pipeline

An end-to-end Machine Learning production-grade pipeline built to predict customer churn using telecom data. This project focuses on handling mixed data types, preventing data leakage, and optimizing models using cross-validation.

## 🚀 Business Overview
In the telecom industry, retaining existing customers is far more cost-effective than acquiring new ones. This project aims to build a predictive model that identifies customers at high risk of churning, allowing marketing and retention teams to take proactive measures.

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Framework:** Scikit-Learn
* **Key Components Used:**
  * `ColumnTransformer` for automated, isolated preprocessing of numerical and categorical data.
  * `Pipeline` to encapsulate the preprocessing steps and the model, ensuring zero data leakage during evaluation.
  * `GridSearchCV` for hyperparameter tuning optimized specifically for **Recall** to minimize False Negatives (missing actual churning customers).

## 📊 Pipeline Workflow
1. **Data Cleaning:** Handled missing values in `TotalCharges` by forcing numeric conversion and dropping null rows.
2. **Feature Engineering & Transformation:**
   * **Numerical Features** (`tenure`, `MonthlyCharges`, `TotalCharges`): Scaled using `StandardScaler`.
   * **Categorical Features** (16 variables including `Contract`, `InternetService`, etc.): Encoded using `OneHotEncoder` handling unknown categories gracefully.
3. **Model Training:** Utilized a `RandomForestClassifier` optimized through 5-fold Stratified Cross-Validation.

## 📈 Hyperparameter Tuning & Results
The model was optimized using `GridSearchCV` with the primary metric set to `Recall (Macro)` to capture churners accurately.

### Best Parameters Found:
* **`n_estimators`**: 150
* **`max_depth`**: 7

### Model Performance (Sample Report):

| Metric | Class (Non-Churn) | Class (Churn) | Macro Avg |
| :--- | :---: | :---: | :---: |
| **Precision** | 0.84 | 0.65 | 0.75 |
| **Recall** | 0.90 | 0.55 | **0.73** |
| **F1-Score** | 0.87 | 0.60 | 0.74 |

*Note: The model strikes a robust balance, prioritizing high recall for the churn class to ensure potential lost customers are flagged effectively.*

## 💻 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/youssefahmed-eng-svg/Customer-Churn-Prediction-Pipeline.git](https://github.com/youssefahmed-eng-svg/Customer-Churn-Prediction-Pipeline.git)
   
   
