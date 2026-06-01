import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================
# Load Dataset
# ==========================
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(df.head())
print(df.shape)
#df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================
# Missing Values
# ==========================
print(df.isnull().sum())

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop("Cabin", axis=1, inplace=True)

# ==========================
# Remove Duplicates
# ==========================
df.drop_duplicates(inplace=True)

# ==========================
# Remove Irrelevant Features
# ==========================
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

# ==========================
# Encode Categorical Variables
# ==========================
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# ==========================
# Feature Scaling
# ==========================
X = df.drop("Survived", axis=1)
y = df["Survived"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# Models
# ==========================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}

# ==========================
# Evaluation
# ==========================
for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n", "="*50)
    print(name)
    print("="*50)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
