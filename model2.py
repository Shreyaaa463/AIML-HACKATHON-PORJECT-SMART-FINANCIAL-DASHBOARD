# ------------------------------
# model2.py – CREDIT CARD MODULE (FINAL VERSION)
# ------------------------------

import pandas as pd
import numpy as np
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


# ------------------------------
# LOAD MERGED DATA
# ------------------------------

df = pd.read_csv("Credit_card_final.csv")
print("Dataset Loaded:", df.shape)


# ------------------------------
# PREPROCESSING (FIXED)
# ------------------------------

# Drop ID column
df = df.drop("Ind_ID", axis=1)

# 1️⃣ Handle Missing Values
for col in df.columns:
    if df[col].dtype == "object":  # categorical
        df[col] = df[col].fillna(df[col].mode()[0])
    else:  # numeric
        df[col] = df[col].fillna(df[col].median())

# 2️⃣ Shuffle Dataset BEFORE anything
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3️⃣ Split features & target
X = df.drop("label", axis=1)
y = df["label"]

# 4️⃣ Encode categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
num_cols = X.select_dtypes(exclude=['object']).columns

label_encoder = LabelEncoder()
for col in cat_cols:
    X[col] = label_encoder.fit_transform(X[col].astype(str))

# 5️⃣ Scale only numeric columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save scaler for deployment
joblib.dump(scaler, "credit_scaler.pkl")


# 6️⃣ TRAIN–TEST SPLIT AFTER SHUFFLING
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=True
)

print("Preprocessing complete!")
print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)


# ------------------------------
# MODELS
# ------------------------------

models = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
models["Logistic Regression"] = lr

# 2. Decision Tree
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    criterion='entropy',
    random_state=42
)
dt.fit(X_train, y_train)
models["Decision Tree"] = dt

# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
)
rf.fit(X_train, y_train)
models["Random Forest"] = rf

# 4. CatBoost
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    verbose=False
)
cat.fit(X_train, y_train)
models["CatBoost"] = cat


# ------------------------------
# METRICS + SAVE BEST MODEL
# ------------------------------

metrics = {}
best_model_name = None
best_acc = 0

for name, model in models.items():

    pred = model.predict(X_val)

    acc = accuracy_score(y_val, pred)
    prec = precision_score(y_val, pred)
    rec = recall_score(y_val, pred)
    f1 = f1_score(y_val, pred)

    metrics[name] = {
        "model": name,
        "accuracy": round(acc, 3),
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3)
    }

    if acc > best_acc:
        best_acc = acc
        best_model_name = name

# Save metrics JSON
with open("credit_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save BEST model
joblib.dump(models[best_model_name], "model_credit.pkl")

print("Best Model:", best_model_name)


# ------------------------------
# VISUALIZATIONS
# ------------------------------

from sklearn.metrics import confusion_matrix

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_val)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Credit Module - Confusion Matrix")
plt.savefig("static/credit_confusion_matrix.png")
plt.close()

# Bar chart comparison
names = list(metrics.keys())
accs = [metrics[m]["accuracy"] for m in names]

plt.figure(figsize=(7,5))
plt.bar(names, accs)
plt.xticks(rotation=15)
plt.title("Credit Module - Model Accuracy Comparison")
plt.savefig("static/credit_model_comparison.png")
plt.close()

print("All credit module outputs saved!")