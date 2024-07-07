import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load the stored model
model_filename = "RF.joblib"
model = joblib.load(model_filename)

# Function to preprocess and predict habitability
def predict_habitability(new_data):
    # Define the indices of the selected parameters (excluding the planet name for training)
    stellar_param_indx = (
        15, 42, 45, 49, 52, 58, 61, 64, 76, 87, 90, 93, 96, 99
    )
    
    # Extract the selected parameters
    selected_columns = new_data.iloc[:, list(stellar_param_indx)]
    
    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    selected_columns = imputer.fit_transform(selected_columns)
    
    # Standardize the feature data
    scaler = StandardScaler()
    selected_columns = scaler.fit_transform(selected_columns)
    
    # Predict habitability
    predictions = model.predict(selected_columns)
    
    # Store the planet names
    planet_names = new_data.iloc[:, 2]
    
    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({
        'Planet Name': planet_names,
        'Predicted Habitability': predictions
    })
    
    return predictions_df

# Example usage
# Load new data of planets
new_planets_data = pd.read_csv('newplanets.csv')

# Predict habitability
predicted_habitability = predict_habitability(new_planets_data)

# Print the predictions
print(predicted_habitability)
