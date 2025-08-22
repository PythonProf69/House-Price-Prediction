import pandas as pd
import numpy as np
import joblib

best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

input_data = pd.DataFrame([
    {
        'longitude': -122.23,
        'latitude': 37.88,
        'housing_median_age': 41,
        'total_rooms': 880,
        'total_bedrooms': 129,
        'population': 322,
        'households': 126,
        'median_income': 8.3252,
        'ocean_proximity': 'NEAR BAY'
    }
])

input_data = pd.get_dummies(input_data)

input_data = input_data.reindex(columns=model_columns, fill_value=0)

std_data = scaler.transform(input_data)

prediction = best_model.predict(std_data)

print("The predicted house price is:", prediction[0])

