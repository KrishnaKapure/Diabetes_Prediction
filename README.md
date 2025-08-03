# Diabetes_Prediction


🩺 Diabetes Prediction Using Machine Learning
This project builds a machine learning model to detect diabetes based on patient health metrics using algorithms like Logistic Regression, Random Forest, and SVM, and provides a desktop GUI (Tkinter) to predict diabetes in new patients.

📌 Features
📊 Data preprocessing (missing values, feature scaling)

⚙️ Models: Logistic Regression, Random Forest, SVM

🧠 Auto-selection of best model based on ROC-AUC score

📈 Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC

🖼️ Feature importance and ROC curve visualizations

🖥️ Desktop GUI (Tkinter) to predict diabetes from user input

💾 Model and scaler saved using joblib

🗂️ Dataset
The model uses patient data with the following features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

Diabetes Pedigree Function

Age

Family History (Yes/No)

Outcome (Target: 0 = No diabetes, 1 = Diabetes)

📁 Dataset file: diabetes.csv

🧪 Project Structure

📁 DiabetesPrediction/
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
├── diabetes_prediction.py    ← Main script
├── feature_importance.png    ← Saved visualization
├── roc_curve.png             ← Saved ROC curve
└── README.md
⚙️ How to Run the Project
Install dependencies

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Run the program

bash
Copy
Edit
python diabetes_prediction.py
This will:

Preprocess the data

Train and evaluate models

Save the best model and scaler

Generate visualizations

Launch the GUI for predictions

🖥️ GUI – Predict Diabetes
Once the program runs, a window will open to input:

Pregnancies

Glucose level

Blood pressure

Skin thickness

Insulin

BMI

Diabetes pedigree function

Age

Family history (Yes/No)

Click "Predict" to receive a result:
“Diabetic” or “Not Diabetic”


📈 Sample Visualizations
feature_importance.png: Displays which features most influence prediction.

<img width="800" height="600" alt="feature_importance" src="https://github.com/user-attachments/assets/45e51c63-0750-488a-92a6-48b870dbb469" />
roc_curve.png: ROC curve for best model performance.

📌 Model Evaluation
During execution, the script prints precision, recall, F1-score, and ROC-AUC for all three models. The one with the highest ROC-AUC is chosen and saved.

<img width="600" height="400" alt="roc_curve" src="https://github.com/user-attachments/assets/d90db1dc-7d28-4216-b7ca-9cf620eff4f0" />

🚀 Future Improvements
Add hyperparameter tuning

Use cross-validation

Extend to web app using Streamlit/Flask

Integrate model explainability (e.g., SHAP)
