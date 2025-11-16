from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import json
import os
os.makedirs('static', exist_ok=True)

app = Flask(__name__)

def encode_form(form):
    mapping = {
        'Male': 1, 'Female': 2, 'Yes': 1, 'No': 2,
        'Graduate': 1, 'Not Graduate': 2,
        'Urban': 3, 'Semiurban': 2, 'Rural': 1,
        'Y': 1, 'N': 0, '3+': 3
    }
    for k in form:
        if form[k] in mapping:
            form[k] = mapping[form[k]]
    return form

# Load credit card model and scaler
credit_model = joblib.load('model_credit.pkl')
credit_scaler = joblib.load('credit_scaler.pkl')

#for fraud.
fraud_model = joblib.load("fraud_model.pkl")
fraud_scaler = joblib.load("fraud_scaler.pkl")

# Load metrics
with open('credit_metrics.json', 'r') as f:
    credit_metrics = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])    #for loan
def predict():
    try:
        data = request.form.to_dict()
        data = encode_form(data)

        model_name = data.pop('model_type')
        model = joblib.load(f"model_{model_name}.pkl")

        numeric_fields = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome',
                          'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
        for f in numeric_fields:
            if f in data and data[f] != '':
                data[f] = float(data[f])
            else:
                data[f] = 0.0

        row = pd.DataFrame([data])

        # CatBoost expects categorical features as int or string
        if model_name == 'cat':
            cat_features = ['Gender', 'Married', 'Dependents', 'Education', 
                            'Self_Employed', 'Property_Area']
            for col in cat_features:
                row[col] = row[col].astype(str)

        pred = int(model.predict(row)[0])

        # Load metrics
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)

        # Map model name
        name_map = {'rf': 'Random Forest', 'dt': 'Decision Tree', 
                    'lr': 'Logistic Regression', 'cat':'CatBoost'}
        model_metrics = metrics.get(name_map.get(model_name, ''), {})

        return jsonify({'prediction': pred, 'metrics': model_metrics})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)})


@app.route('/predict_credit', methods=['POST'])
def predict_credit():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Define the correct column order from training
        columns = ['GENDER', 'Car_Owner', 'Propert_Owner', 'CHILDREN', 'Annual_income',
                  'Type_Income', 'EDUCATION', 'Marital_status', 'Housing_type',
                  'Birthday_count', 'Employed_days', 'Mobile_phone', 'Work_Phone',
                  'Phone', 'EMAIL_ID', 'Type_Occupation', 'Family_Members']
        
        # Create DataFrame with correct column order
        input_data = pd.DataFrame([data], columns=columns)
        
        # Define numeric and categorical columns
        numeric_cols = ['CHILDREN', 'Annual_income', 'Birthday_count', 'Employed_days',
                       'Mobile_phone', 'Work_Phone', 'Phone', 'EMAIL_ID', 'Family_Members']
        
        cat_cols = ['GENDER', 'Car_Owner', 'Propert_Owner', 'Type_Income', 'EDUCATION',
                   'Marital_status', 'Housing_type', 'Type_Occupation']
        
        # Convert numeric columns
        for col in numeric_cols:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Handle missing values
        for col in cat_cols:
            if input_data[col].isna().any() or input_data[col].iloc[0] == '':
                input_data[col] = input_data[col].fillna('Unknown')
        
        for col in numeric_cols:
            if input_data[col].isna().any():
                input_data[col] = input_data[col].fillna(0)
        
        # Encode categorical columns using LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in cat_cols:
            input_data[col] = le.fit_transform(input_data[col].astype(str))
        
        # Scale numeric columns
        input_data[numeric_cols] = credit_scaler.transform(input_data[numeric_cols])
        
        # Make prediction
        prediction = int(credit_model.predict(input_data)[0])
        
        # Get best model metrics
        best_model_name = max(credit_metrics, key=lambda x: credit_metrics[x]['accuracy'])
        metrics = credit_metrics[best_model_name]

        return jsonify({
            'prediction': prediction,
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({'error': str(e)})






@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    try:
        data = request.json
        
        input_values = [
            float(data["TransactionAmount"]),
            float(data["CustomerAge"]),
            float(data["TransactionDuration"]),
            float(data["LoginAttempts"]),
            float(data["AccountBalance"])
        ]

        arr = np.array(input_values).reshape(1, -1)

        scaled = fraud_scaler.transform(arr)

        pred = fraud_model.predict(scaled)[0]   # -1 = anomaly, 1 = normal
        score = fraud_model.decision_function(scaled)[0]

        result = "FRAUD" if pred == -1 else "NORMAL"

        return {
            "prediction": result,
            "score": float(score)
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == '__main__':
    app.run(debug=True)



