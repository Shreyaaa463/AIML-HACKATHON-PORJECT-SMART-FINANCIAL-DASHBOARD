import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings("ignore")


def preprocess(df):
    """Fill missing values with mode for categorical and mean for numeric columns."""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
    return df

def remove_outliers(df, columns):
    """Remove outliers using IQR method for specified columns."""
    df_clean = df.copy()
   
    for col in columns:
        if col in df_clean.columns and df_clean[col].dtype in ['int64', 'float64']:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
           
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
           
            # Count outliers before removal
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
           
            # Remove outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
           
            print(f"  {col}: Removed {outliers_count} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")
   
    return df_clean

def encode(df):
    """Encode categorical variables using predefined mappings."""
    mapping = {
        'Male': 1, 'Female': 2,
        'Yes': 1, 'No': 2,
        'Graduate': 1, 'Not Graduate': 2,
        'Urban': 3, 'Semiurban': 2, 'Rural': 1,
        'Y': 1, 'N': 0
    }

    for col in df.select_dtypes('object').columns:
        df[col] = df[col].map(lambda x: mapping.get(x, x))
    
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3)
        df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce').fillna(0).astype(int)
    
    return df

print("Loading data...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocess and encode
train = preprocess(train)
train = encode(train)
test = preprocess(test)
test = encode(test)

print("\nRemoving outliers from training data...")
outlier_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
train_before = len(train)
train = remove_outliers(train, outlier_columns)
print(f"Training samples: {train_before} → {len(train)} (removed {train_before - len(train)} rows)\n")

# Drop unnecessary columns
X = train.drop(['Loan_ID', 'Loan_Status'], axis=1, errors='ignore')
y = train['Loan_Status']

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale only for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ---------------------- Train CatBoost ----------------------
print("Training CatBoost Classifier...")
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                    'Self_Employed', 'Property_Area']

cat = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=8, 
    l2_leaf_reg=3, 
    random_seed=42,
    verbose=100, 
    eval_metric='Accuracy'
)
cat.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_val, y_val))
joblib.dump(cat, 'model_cat.pkl')
pred_cat = cat.predict(X_val)
print("CatBoost Train Accuracy:", cat.score(X_train, y_train))
print("CatBoost Validation Accuracy:", cat.score(X_val, y_val))

# ---------------------- Train Random Forest ----------------------
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)
joblib.dump(rf, 'model_rf.pkl')
pred_rf = rf.predict(X_val)
print("Random Forest Train Accuracy:", rf.score(X_train, y_train))
print("Random Forest Validation Accuracy:", rf.score(X_val, y_val))


# ---------------------- Train Decision Tree ----------------------
print("Training Decision Tree...") 
dt = DecisionTreeClassifier(
    criterion='entropy',        # entropy for information gain
    max_depth=5,                # limit depth to reduce overfitting
    min_samples_split=8,        # require more samples to split
    min_samples_leaf=4,         # minimum samples per leaf
    max_features='sqrt',        # consider subset of features at each split
    class_weight='balanced',    # handles class imbalance if any
    random_state=42 
)
dt.fit(X_train, y_train)
joblib.dump(dt, 'model_dt.pkl')   

pred_dt = dt.predict(X_val)

print("Decision Tree Train Accuracy:", dt.score(X_train, y_train))
print("Decision Tree Validation Accuracy:", dt.score(X_val, y_val))



# ---------------------- Train Logistic Regression ----------------------
print("Training Logistic Regression...")
lr = LogisticRegression(
    C=1.5,
    solver='saga',
    penalty='l2',
    max_iter=3000,
    random_state=42
)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, 'model_lr.pkl')
pred_lr = lr.predict(X_val_scaled)
print("Logistic Regression Train Accuracy:", lr.score(X_train_scaled, y_train))
print("Logistic Regression Validation Accuracy:", lr.score(X_val_scaled, y_val))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Store accuracies for all models
model_names = ['Logistic Regression', 'Random Forest', 'Decision Tree', 'CatBoost']
accuracies = [
    lr.score(X_val_scaled, y_val),
    rf.score(X_val, y_val),
    dt.score(X_val, y_val),
    cat.score(X_val, y_val)
]

# Save bar chart comparing accuracies
plt.figure(figsize=(6,4))
plt.bar(model_names, accuracies)
plt.xlabel("Models")
plt.ylabel("Validation Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("static/model_comparison.png")
plt.close()

# --- Confusion Matrix for best model CatBoost ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

best_model = cat 
y_pred_best = best_model.predict(X_val)
cm = confusion_matrix(y_val, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (CatBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/confusion_matrix.png")
plt.close()




# ---------------------- Evaluate Models ----------------------
print("\nEvaluating models...\n")

print(f"Decision Tree Accuracy: {accuracy_score(y_val, pred_dt):.2f}")
print(classification_report(y_val, pred_dt))

print(f"Random Forest Accuracy: {accuracy_score(y_val, pred_rf):.2f}")
print(classification_report(y_val, pred_rf))

print(f"Logistic Regression Accuracy: {accuracy_score(y_val, pred_lr):.2f}")
print(classification_report(y_val, pred_lr))

print(f"CatBoost Accuracy: {accuracy_score(y_val, pred_cat):.2f}")
print(classification_report(y_val, pred_cat))

from sklearn.metrics import precision_score, recall_score, f1_score
import json

metrics = {}

def get_metrics(y_true, y_pred, name):
    return {
        'model': name,
        'accuracy': round(accuracy_score(y_true, y_pred), 3),
        'precision': round(precision_score(y_true, y_pred, pos_label=1), 3),
        'recall': round(recall_score(y_true, y_pred, pos_label=1), 3),
        'f1': round(f1_score(y_true, y_pred, pos_label=1), 3)
    }

metrics['Decision Tree'] = get_metrics(y_val, pred_dt, 'Decision Tree')
metrics['Random Forest'] = get_metrics(y_val, pred_rf, 'Random Forest')
metrics['Logistic Regression'] = get_metrics(y_val, pred_lr, 'Logistic Regression')
metrics['CatBoost'] = get_metrics(y_val, pred_cat, 'CatBoost')

with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("\n✅ Model metrics saved to model_metrics.json")
