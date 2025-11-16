import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import joblib
import os

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# Load CSV
df = pd.read_csv("fraud_detection_data.csv")

# Select ONLY numerical columns
num_cols = [
    "TransactionAmount",
    "CustomerAge",
    "TransactionDuration",
    "LoginAttempts",
    "AccountBalance"
]

data = df[num_cols].copy()

# Handle missing values
data = data.fillna(data.mean())

# -----------------------------
# 1️⃣ CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix - Fraud Dataset")
plt.tight_layout()
plt.savefig("static/fraud_corr_matrix.png")
plt.close()

# -----------------------------
# 2️⃣ DISTRIBUTION OF FEATURES
# -----------------------------
plt.figure(figsize=(12, 8))
data.hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions - Fraud Dataset")
plt.tight_layout()
plt.savefig("static/fraud_feature_distributions.png")
plt.close()

# -----------------------------
# 3️⃣ BOX PLOTS FOR OUTLIERS
# -----------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title("Boxplots of Numerical Features - Fraud Dataset")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/fraud_boxplots.png")
plt.close()

# -----------------------------
# Train Isolation Forest
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

model = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # Assume ~5% anomalies
    random_state=42
)

model.fit(scaled_data)

# -----------------------------
# 4️⃣ ANOMALY SCORE DISTRIBUTION
# -----------------------------
scores = model.decision_function(scaled_data)

plt.figure(figsize=(10, 5))
sns.histplot(scores, bins=40, kde=True)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("static/fraud_anomaly_scores.png")
plt.close()

# -----------------------------
# 5️⃣ PCA Scatter Plot (Normal vs Anomalies)
# -----------------------------
preds = model.predict(scaled_data)   # -1 = anomaly, 1 = normal

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 7))
plt.scatter(
    pca_data[preds == 1, 0],
    pca_data[preds == 1, 1],
    label="Normal",
    alpha=0.6
)
plt.scatter(
    pca_data[preds == -1, 0],
    pca_data[preds == -1, 1],
    label="Anomaly",
    alpha=0.9
)
plt.title("PCA Scatter Plot - Normal vs Anomalous Transactions")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("static/fraud_anomaly_scatter.png")
plt.close()

# -----------------------------
# Save model + scaler
# -----------------------------
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "fraud_scaler.pkl")

print("Model training complete! Saved model, scaler & ALL visualizations in /static/")