# Customer Churn Prediction - Simple Data Science Project

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# 1. Load dataset
data = pd.read_csv("customer_churn.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset information:")
print(data.info())

# 2. Clean missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Fill numeric missing values with median
data["monthly_charges"] = data["monthly_charges"].fillna(
    data["monthly_charges"].median()
)

# Fill text missing values with most common value
data["payment_method"] = data["payment_method"].fillna(
    data["payment_method"].mode()[0]
)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# 3. Simple EDA
print("\nChurn count:")
print(data["churn"].value_counts())

plt.figure(figsize=(6, 4))
data["churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
data.groupby("churn")["tenure"].mean().plot(kind="bar")
plt.title("Average Tenure by Churn")
plt.xlabel("Churn")
plt.ylabel("Average Tenure")
plt.tight_layout()
plt.show()

# 4. Convert Yes/No columns to numbers
binary_columns = [
    "tech_support",
    "senior_citizen",
    "churn"
]

encoder = LabelEncoder()

for col in binary_columns:
    data[col] = encoder.fit_transform(data[col])

# 5. Convert categorical columns into dummy variables
data = pd.get_dummies(
    data,
    columns=["contract", "internet_service", "payment_method"],
    drop_first=True
)

# 6. Split input and output
X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Logistic Regression
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X_train, y_train)

logistic_pred = logistic.predict(X_test)
logistic_prob = logistic.predict_proba(X_test)[:, 1]

# 8. Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)
forest_prob = forest.predict_proba(X_test)[:, 1]

# 9. Evaluate models
print("\n--- Logistic Regression ---")
print("Accuracy:", round(accuracy_score(y_test, logistic_pred), 3))
print("F1 Score:", round(f1_score(y_test, logistic_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, logistic_prob), 3))
print(classification_report(y_test, logistic_pred))

print("\n--- Random Forest ---")
print("Accuracy:", round(accuracy_score(y_test, forest_pred), 3))
print("F1 Score:", round(f1_score(y_test, forest_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, forest_prob), 3))
print(classification_report(y_test, forest_pred))

# 10. Feature importance
importance = pd.Series(
    forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop important features:")
print(importance.head(10))

plt.figure(figsize=(8, 5))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

print("\nProject completed successfully!")
