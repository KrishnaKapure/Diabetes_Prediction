# Importing necessary libraries for data manipulation, visualization, GUI, and ML
import pandas as pd  # For data handling using DataFrames
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns  # For advanced visualizations
import tkinter as tk  # For creating GUI applications
from tkinter import messagebox  # For pop-up message boxes in GUI
import joblib  # For saving/loading models and objects

# Importing scikit-learn modules for ML and preprocessing
from sklearn.model_selection import train_test_split  # For splitting data into train and test
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import (  # Evaluation metrics
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, roc_curve
)

# -------- Global variables --------------
# These will be used throughout the script
scaler = None
best_model = None
X = None
X_test = None
y_test = None

# ---------- Data Preprocessing Function --------------
def Load_PreprocessData():
    global X, X_test, y_test, scaler  # Access global variables

    # Load the dataset from CSV
    df = pd.read_csv('diabetes.csv')

    # Replace 0s with NaN for specific columns where 0 is invalid
    columns_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_fix] = df[columns_to_fix].replace(0, np.nan)

    # Impute missing values with mean
    imputer = SimpleImputer(strategy='mean')
    df[columns_to_fix] = imputer.fit_transform(df[columns_to_fix])

    # Handle Family History feature (add if missing, convert if needed)
    if 'Family History' not in df.columns:
        df['Family History'] = 0  # If not present, add column with 0
    elif df['Family History'].dtype == object:
        df['Family History'] = df['Family History'].map({'Yes': 1, 'No': 0})  # Convert Yes/No to 1/0

    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test_local, y_train, y_test_local = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Save test data in global variables
    X_test = X_test_local
    y_test = y_test_local
    return X_train, y_train  # Return training data

# ------------- Model Training Function --------------
def Train_Model(X_train, y_train):
    global best_model  # Access global variable

    # Dictionary of models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)  # Enable probability output for ROC
    }

    best_auc = 0  # Variable to store best AUC score

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nModel: {name}")
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Predict labels
        y_proba = model.predict_proba(X_test)[:, 1]  # Predict probabilities for ROC

        # Print classification report
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)  # Compute AUC
        print("ROC-AUC Score:", auc)

        # Update best model if AUC is better
        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save best model and scaler to files
    joblib.dump(best_model, 'diabetes_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

# ------------- Feature Importance and ROC Curve Visualization --------------
def Visualizations():
    # Plot feature importance (only works for Random Forest)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        features = X.columns
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances, y=features)
        plt.title("Feature Importance - Random Forest")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.show()

    # Plot ROC curve
    y_proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()

# --------------- GUI (User Interface) ---------------
def launch_gui():
    try:
        # Load trained model and scaler
        model = joblib.load("diabetes_model.pkl")
        scaler_local = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Model or scaler file not found.")
        return

    # Prediction function triggered on button click
    def predict_diabetes():
        try:
            # Read input values from text fields and convert to float
            values = [
                float(entry_pregnancies.get()),
                float(entry_glucose.get()),
                float(entry_bp.get()),
                float(entry_skin.get()),
                float(entry_insulin.get()),
                float(entry_bmi.get()),
                float(entry_dpf.get()),
                float(entry_age.get()),
                1 if entry_family.get().strip().lower() == "yes" else 0
            ]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

        # Scale the input values using the saved scaler
        values_scaled = scaler_local.transform([values])

        # Predict using the trained model
        prediction = model.predict(values_scaled)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        # Show result in a popup
        messagebox.showinfo("Prediction Result", f"The person is likely: {result}")

    # ------------------ GUI Layout ------------------
    app = tk.Tk()  # Create the main window
    app.title("Diabetes Prediction")  # Window title
    app.geometry("400x600")  # Set window size

    # Define the input fields and their variable names
    fields = [
        ("Pregnancies", "entry_pregnancies"),
        ("Glucose Level", "entry_glucose"),
        ("Blood Pressure", "entry_bp"),
        ("Skin Thickness", "entry_skin"),
        ("Insulin", "entry_insulin"),
        ("BMI", "entry_bmi"),
        ("Diabetes Pedigree Function", "entry_dpf"),
        ("Age", "entry_age"),
        ("Family History (Yes/No)", "entry_family")
    ]

    # Create input labels and entry boxes
    for label_text, var_name in fields:
        tk.Label(app, text=label_text).pack(pady=(10, 0))
        globals()[var_name] = tk.Entry(app)
        globals()[var_name].pack()

    # Predict button
    tk.Button(app, text="Predict", command=predict_diabetes, bg="green", fg="white").pack(pady=20)

    app.mainloop()  # Run the GUI loop

# ------------- Main Function to Run Everything --------------
def main():
    # Step 1: Load and preprocess the data
    X_train, y_train = Load_PreprocessData()

    # Step 2: Train the model
    Train_Model(X_train, y_train)

    # Step 3: Visualize results
    Visualizations()

    # Step 4: Launch the GUI
    launch_gui()

# Run main only if the script is executed directly
if __name__ == "__main__":
    main()
