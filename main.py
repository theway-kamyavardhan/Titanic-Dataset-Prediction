# Titanic Survival Prediction using Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# =========================== Load Training Data ===========================

print("\n==== Loading Train Data ====\n")
train_data = pd.read_csv("train.csv")
print(train_data.head())

print("\n==== Train Data Columns ====\n")
print(train_data.columns)

print("\n==== Missing Values in Train Data ====\n")
print(train_data.isna().sum())

print("\n==== Age Column Before Filling Nulls ====\n")
print(train_data['Age'])

# =========================== Data Cleaning (Train) ===========================

# Fill missing Age values with median
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())

print("\n==== Age Column After Filling Nulls ====\n")
print(train_data['Age'])

# Drop Cabin column (not useful due to many nulls)
train_data = train_data.drop(columns=['Cabin'])

# Fill missing Embarked values with "S"
train_data['Embarked'] = train_data['Embarked'].fillna("S")

print("\n==== Missing Values After Cleaning Train Data ====\n")
print(train_data.isna().sum())

# =========================== Encoding Categorical Columns ===========================

label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()

train_data['Sex'] = label_encoder_sex.fit_transform(train_data['Sex'])
train_data['Embarked'] = label_encoder_embarked.fit_transform(train_data['Embarked'])

print("\n==== Train Data After Encoding ====\n")
print(train_data.head())

# =========================== Feature Selection ===========================

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = train_data[features]
Y = train_data[['Survived']]

print("\n==== Feature Columns (X) ====\n")
print(X.head())

print("\n==== Target Column (Y) ====\n")
print(Y.head())

# =========================== Scaling Features ===========================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\n==== Scaled Features Preview ====\n")
print(X_scaled[:5])

# =========================== Train-Test Split ===========================

X_train, X_val, Y_train, Y_val = train_test_split(
    X_scaled, Y, random_state=42, test_size=0.3, stratify=Y
)

# =========================== Model Training ===========================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# =========================== Model Prediction ===========================

y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)

# =========================== Model Evaluation ===========================

accuracy = accuracy_score(Y_val, y_pred)
conf_matrix = confusion_matrix(Y_val, y_pred)
f1 = f1_score(Y_val, y_pred)

print("\n==== Model Accuracy ====\n")
print(accuracy)

print("\n==== Confusion Matrix ====\n")
print(conf_matrix)

print("\n==== F1 Score ====\n")
print(f1)

# =========================== Process Test Data ===========================

print("\n==== Loading Test Data ====\n")
test_data = pd.read_csv("test.csv")
print(test_data.head())

print("\n==== Missing Values in Test Data ====\n")
print(test_data.isna().sum())

# Fill missing Age and Fare with median
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# Drop Cabin column
test_data = test_data.drop(columns=['Cabin'])

# Encode categorical columns (using train encoders)
test_data['Sex'] = label_encoder_sex.transform(test_data['Sex'])
test_data['Embarked'] = label_encoder_embarked.transform(test_data['Embarked'])

# Select Features and Scale
X_test_data = test_data[features]
X_test_scaled = scaler.transform(X_test_data)

# =========================== Final Predictions on Test Set ===========================

y_test_predictions = model.predict(X_test_scaled)

print("\n==== Final Predictions on Test Data ====\n")
print(y_test_predictions)

# =========================== Generate Submission File ===========================

submission_file = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_test_predictions
})

submission_file.to_csv("submission.csv", index=False)
print("\n==== Submission File 'submission.csv' Created Successfully ====\n")

# =========================== Confusion Matrix Plot ===========================

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure for README or reporting
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("\n==== Confusion Matrix Plot Saved as 'confusion_matrix.png' ====\n")
plt.close()  # Optional: Close the plot to avoid display during batch runs
