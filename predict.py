import pandas as pd
from sklearn.preprocessing import StandardScaler
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

# Load new data for prediction
predict_data = pd.read_csv('newplanets.csv')

# Ensure kepoi_name is present
kepoi_name = predict_data['kepoi_name']

# Select specific features
selected_features = list(planetary_stellar_parameter_cols_dict.keys())
predict_data = predict_data[selected_features]

# Drop rows with missing values
predict_data.dropna(inplace=True)

# Scale the features
X_predict = scaler.transform(predict_data)

# Predict using the loaded model
predictions = svm.predict(X_predict)

# Print the predictions
prediction_results = pd.DataFrame({
    'kepoi_name': kepoi_name,
    'habitable': predictions
})

# Convert the prediction results to a human-readable format
prediction_results['habitable'] = prediction_results['habitable'].map({1: 'Habitable', 0: 'Non-Habitable'})

print(prediction_results)