import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load the saved model and scaler
svm = joblib.load('svm_habitability_model.pkl')
scaler = joblib.load('scaler.pkl')

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

# Shuffle the dataset and take a subset for new test data
test_data = planets.sample(frac=0.1, random_state=42).reset_index(drop=True)
test_names = kepoi_name.iloc[test_data.index].reset_index(drop=True)

# Separate features and target
X_new_test = test_data[selected_features]
y_new_test = test_data['target']

# Scale the features
X_new_test = scaler.transform(X_new_test)

# Predict using the loaded model
y_new_pred = svm.predict(X_new_test)
accuracy = accuracy_score(y_new_test, y_new_pred)

# Print evaluation metrics
print(f"Accuracy of the model on new test data is: {accuracy}")
print(classification_report(y_new_test, y_new_pred))
print(confusion_matrix(y_new_test, y_new_pred))

# Print names of planets predicted to be habitable
new_habitable_planet_names = test_names[y_new_pred == 1]
print("Planets predicted to be habitable:")
print(new_habitable_planet_names)
