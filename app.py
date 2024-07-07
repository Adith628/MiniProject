import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy of the model is: {accuracy}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Print names of planets predicted to be habitable
habitable_planet_names = kepoi_name.iloc[y_test.index][y_pred == 1]
print("Planets predicted to be habitable:")
print(habitable_planet_names)

# Save the model and scaler
joblib.dump(svm, 'svm_habitability_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
