📊 Smart Financial Dashboard

An AI-driven financial analytics system integrating loan approval prediction, credit card approval prediction, and bank fraud anomaly detection into a unified dashboard.

🎯 Project Objectives

To design and develop an intelligent dashboard that automates key financial decision-making tasks.

To build robust ML models for evaluating loan approval and credit card approval.

To detect fraudulent or abnormal bank transactions using unsupervised anomaly detection.

To provide visual insights through graphs, correlation heatmaps, PCA plots, and confusion matrices.

To deploy an interactive UI using HTML, CSS, and JavaScript for real-time predictions.

📂 Dataset Details
1. Loan Approval Dataset:

Contains applicant details such as income, employment status, credit history, loan amount, dependents, and more.
(Data sourced from publicly available loan approval datasets.)

2. Credit Card Approval Dataset:

Includes demographic and financial attributes:

Ind_ID, Gender, Car_Owner, Property_Owner

Children, Annual Income, Type of Income

Education, Marital Status, Housing Type

Birthday_Count, Employed_Days

Mobile/Work Phone indicators

Email_ID, Occupation Type, Family Members

Label (Approved / Not Approved)

3. Bank Fraud Detection Dataset:

Numerical features used:

Transaction Amount

Customer Age

Transaction Duration

Login Attempts

Account Balance

PCA-reduced features used for anomaly visualization (Component 1 & 2).

🧠 Algorithms & Models Used
Loan Approval Prediction

Models implemented:

CatBoost (Best Model – 89% accuracy)

Random Forest

XGBoost

Logistic Regression

Performed hyperparameter tuning and model comparison.

Credit Card Approval Prediction

Models implemented:

CatBoost

Random Forest

Decision Tree

Logistic Regression

Best model is automatically saved and used by the dashboard for prediction.

Fraud Detection (Anomaly Detection)

Isolation Forest

PCA for scatter visualization of anomalous vs. normal transactions

Anomaly score distribution plotted

Data Preprocessing

Missing value handling

Outlier removal (Boxplots)

Scaling & encoding

Correlation heatmaps

Feature distribution analysis

📈 Results
Loan Approval Prediction

Best Model: CatBoost

Accuracy: ~89%

Visualizations:

Confusion Matrix

Correlation Matrix

Feature Importance Plot

Credit Card Approval Prediction

Best model dynamically selected and saved

Visualizations:

Confusion Matrix

Model Comparison (Bar Chart)

Correlation Heatmap

Fraud Detection

Isolation Forest identified anomalies effectively

Visualizations:

PCA 2-D Scatter Plot

Count vs. Anomaly Score Plot

Exploratory Data Analysis

Box Plots (outliers)

Feature Distribution Plots

Correlation Heatmaps

All visual outputs are displayed in the dashboard.

🧪 Conclusion

The Smart Financial Dashboard successfully integrates predictive analytics and anomaly detection into one unified platform. The CatBoost model consistently delivered the best performance for both loan and credit card approval tasks, while Isolation Forest effectively identified suspicious transactions in the banking dataset. The system demonstrates strong potential for assisting financial institutions with automated decision-making and fraud prevention.

🚀 Future Scope

Integrate deep learning models for improved fraud detection

Expand the dashboard to include customer segmentation and credit scoring

Deploy using Flask/Django + React for production-level use

Connect with real-time transactional APIs

Add explainable AI (SHAP values) for transparent decision-making

📚 References

CatBoost Documentation – https://catboost.ai

scikit-learn Documentation – https://scikit-learn.org

XGBoost Documentation – https://xgboost.readthedocs.io

Public datasets from Kaggle & UCI Machine Learning Repository

Research papers on Credit Scoring & Anomaly Detection (IEEE, Springer)
