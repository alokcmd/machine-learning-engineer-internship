# Customer Churn Prediction Using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load Dataset

df = pd.read_csv("data/customer_churn.csv")

print("Dataset Loaded Successfully")
print(df.head())

# 2. Data Understanding

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Shape:")
print(df.shape)

# 3. Data Cleaning

df = df.drop_duplicates()

# Remove customer ID
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert TotalCharges into numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

# Remove rows where target is missing
df = df.dropna(subset=["Churn"])

# 4. Exploratory Data Analysis

sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

# 5. Separate Features and Target

X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# 6. Identify Numerical and Categorical Columns

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 7. Preprocessing

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# 8. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Train Logistic Regression

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

# 10. Train Random Forest

random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

random_forest.fit(X_train, y_train)

# 11. Model Evaluation

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest
}

best_model = None
best_auc = -1

for name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    print("\nModel:", name)
    print("Accuracy:", accuracy)
    print("ROC-AUC:", auc)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if auc > best_auc:
        best_auc = auc
        best_model = model

# 12. Save Best Model

joblib.dump(best_model, "customer_churn_model.pkl")

print("\nBest Model Saved Successfully!")