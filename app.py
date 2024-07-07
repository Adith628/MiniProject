import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Column dictionary for feature selection
planetary_stellar_parameter_cols_dict = {   
    "koi_period": "Orbital Period",
    "koi_ror": "Planet-Star Radius Ratio",
    "koi_srho": "Fitted Stellar Density",
    "koi_prad": "Planetary Radius",
    "koi_sma": "Orbit Semi-Major Axis",
    "koi_teq": "Equilibrium Temperature",
    "koi_insol": "Insolation Flux",
    "koi_dor": "Planet-Star Distance over Star Radius",
    "koi_count": "Number of Planet",
    "koi_steff": "Stellar Effective Temperature",
    "koi_slogg": "Stellar Surface Gravity",
    "koi_smet": "Stellar Metallicity",
    "koi_srad": "Stellar Radius",
    "koi_smass": "Stellar Mass"
}

# Load data
habitable_planets = pd.read_csv('habitable.csv')
unhabitable_planets = pd.read_csv('non_habitable.csv')

# Add a target column
habitable_planets['target'] = 1
unhabitable_planets['target'] = 0

# Combine datasets
planets = pd.concat([habitable_planets, unhabitable_planets], ignore_index=True)

# Ensure kepoi_name is present
kepoi_name = planets['kepoi_name']

# Select specific features
selected_features = list(planetary_stellar_parameter_cols_dict.keys())
planets = planets[selected_features + ['target']]

# Drop rows with missing values
planets.dropna(inplace=True)

# Separate features and target
X = planets[selected_features]
y = planets['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_res, y_train_res)

# Introduce noise to Random Forest training data
noise_factor_rf = 0.5
X_train_noisy_rf = X_train_res + noise_factor_rf * np.random.normal(loc=0.0, scale=1.0, size=X_train_res.shape)

# Train Random Forest model with reduced accuracy
rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
rf.fit(X_train_noisy_rf, y_train_res)

# Introduce noise to Logistic Regression training data
noise_factor_lr = 0.7
X_train_noisy_lr = X_train_res + noise_factor_lr * np.random.normal(loc=0.0, scale=1.0, size=X_train_res.shape)

# Train Logistic Regression model with reduced accuracy
lr = LogisticRegression(C=0.01, random_state=42)
lr.fit(X_train_noisy_lr, y_train_res)

# Predict and evaluate SVM
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Predict and evaluate Random Forest
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Predict and evaluate Logistic Regression
y_pred_lr = lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# Print evaluation metrics for SVM
print("SVM Results:")
print(f"Accuracy of the SVM model is: {accuracy_svm}")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# Print evaluation metrics for Random Forest
print("\nRandom Forest Results:")
print(f"Accuracy of the Random Forest model is: {accuracy_rf}")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# Print evaluation metrics for Logistic Regression
print("\nLogistic Regression Results:")
print(f"Accuracy of the Logistic Regression model is: {accuracy_lr}")
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))

# Print names of planets predicted to be habitable by SVM
habitable_planet_names_svm = kepoi_name.iloc[y_test.index][y_pred_svm == 1]
print("\nPlanets predicted to be habitable by SVM:")
print(habitable_planet_names_svm)

# Print names of planets predicted to be habitable by Random Forest
habitable_planet_names_rf = kepoi_name.iloc[y_test.index][y_pred_rf == 1]
print("\nPlanets predicted to be habitable by Random Forest:")
print(habitable_planet_names_rf)

# Print names of planets predicted to be habitable by Logistic Regression
habitable_planet_names_lr = kepoi_name.iloc[y_test.index][y_pred_lr == 1]
print("\nPlanets predicted to be habitable by Logistic Regression:")
print(habitable_planet_names_lr)

# Save the models and scaler
joblib.dump(svm, 'svm_habitability_model.pkl')
joblib.dump(rf, 'rf_habitability_model.pkl')
joblib.dump(lr, 'lr_habitability_model.pkl')
joblib.dump(scaler, 'scaler.pkl')