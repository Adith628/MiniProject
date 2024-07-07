from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from django.views.decorators.csrf import csrf_exempt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svm_habitability_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

svm = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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

@csrf_exempt
def predict_habitability(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        try:
            data = pd.read_csv(file)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

        if 'kepoi_name' not in data.columns:
            return JsonResponse({"error": "'kepoi_name' column is missing from the input file"}, status=400)

        kepoi_name = data['kepoi_name']

        selected_features = list(planetary_stellar_parameter_cols_dict.keys())
        if not all(col in data.columns for col in selected_features):
            return JsonResponse({"error": "Input file does not contain all required features"}, status=400)

        data = data[selected_features]
        data.dropna(inplace=True)

        X_predict = scaler.transform(data)
        predictions = svm.predict(X_predict)

        prediction_results = pd.DataFrame({
            'kepoi_name': kepoi_name,
            'habitable': predictions
        })

        prediction_results['habitable'] = prediction_results['habitable'].map({1: 'Habitable', 0: 'Non-Habitable'})
        results = prediction_results.to_dict(orient='records')

        return JsonResponse(results, safe=False)

    return JsonResponse({"error": "Invalid request method"}, status=405)
