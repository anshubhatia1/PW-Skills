from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Concrete Strength Predictor API")

# Load artifacts
model = joblib.load('final_model.joblib')
imputer = joblib.load('knn_imputer.joblib')
poly_transformer = joblib.load('poly_transformer.joblib')

with open('feature_lists.json', 'r') as f:
    feature_lists = json.load(f)

original_input_columns = feature_lists['original_input_columns']
poly_features_cols = feature_lists['poly_features_cols']
all_engineered_feature_names = feature_lists['all_engineered_feature_names']
selected_feature_names = feature_lists['selected_feature_names']

# Pydantic model for input data validation
class ConcreteInput(BaseModel):
    Cement: float
    Blast_Furnace_Slag: float # Underscore for Pydantic, will match df column after cleaning
    Fly_Ash: float
    Water: float
    Superplasticizer: float
    Coarse_Aggregate: float
    Fine_Aggregate: float
    Age: float

class PredictionOut(BaseModel):
    predicted_strength: float

@app.post("/predict/", response_model=PredictionOut)
async def predict_strength(input_data: ConcreteInput):
    # Convert Pydantic model to dictionary, then to DataFrame
    # Ensure keys match the 'original_input_columns' used for training DataFrame construction
    data_dict = {
        'Cement': input_data.Cement,
        'Blast Furnace Slag': input_data.Blast_Furnace_Slag, # Match original df column names
        'Fly Ash': input_data.Fly_Ash,
        'Water': input_data.Water,
        'Superplasticizer': input_data.Superplasticizer,
        'Coarse Aggregate': input_data.Coarse_Aggregate,
        'Fine Aggregate': input_data.Fine_Aggregate,
        'Age': input_data.Age
    }
    input_df_orig = pd.DataFrame([data_dict], columns=original_input_columns)

    # --- Apply Feature Engineering ---
    X_eng_api = input_df_orig.copy()
    epsilon = 1e-6

    # 1. Ratio Features (ensure column names used here are the cleaned ones)
    X_eng_api['Water_to_Cement'] = input_df_orig['Water'] / (input_df_orig['Cement'] + epsilon)
    X_eng_api['Superplasticizer_to_Cement'] = input_df_orig['Superplasticizer'] / (input_df_orig['Cement'] + epsilon)
    X_eng_api['FlyAsh_to_Cement'] = input_df_orig['Fly Ash'] / (input_df_orig['Cement'] + epsilon)
    X_eng_api['Slag_to_Cement'] = input_df_orig['Blast Furnace Slag'] / (input_df_orig['Cement'] + epsilon)
    X_eng_api['FineAgg_to_CoarseAgg'] = input_df_orig['Fine Aggregate'] / (input_df_orig['Coarse Aggregate'] + epsilon)
    X_eng_api['CoarseAgg_to_Cement'] = input_df_orig['Coarse Aggregate'] / (input_df_orig['Cement'] + epsilon)
    X_eng_api['FineAgg_to_Cement'] = input_df_orig['Fine Aggregate'] / (input_df_orig['Cement'] + epsilon)

    # 2. Polynomial Features
    # Use the loaded poly_transformer (fitted on specific columns during training)
    X_poly_api = poly_transformer.transform(input_df_orig[poly_features_cols])
    poly_feature_names_api = poly_transformer.get_feature_names_out(poly_features_cols)
    X_poly_api_df = pd.DataFrame(X_poly_api, columns=poly_feature_names_api, index=input_df_orig.index)

    # Combine: original features used for ratios + new ratio features + new poly features (minus duplicates)
    X_eng_api = pd.concat([X_eng_api, X_poly_api_df.drop(columns=poly_features_cols)], axis=1)

    # Ensure all engineered columns as per training are present, fill missing with NaN (for imputer)
    # This step is crucial if some features were created that are not directly derivable from the 8 inputs alone
    # (e.g. some complex interaction not covered by poly_transformer on just the basic inputs)
    # For this case, all_engineered_feature_names should be reconstructible.
    # However, to be robust, reindex before imputation.

    # Reorder/select columns to match 'all_engineered_feature_names' before imputation
    # This ensures the imputer sees data in the same format as during its fit method
    X_eng_api_reordered = X_eng_api.reindex(columns=all_engineered_feature_names, fill_value=np.nan)

    # Impute missing values (if any, after feature engineering)
    # The imputer was fit on xtrain (which had all engineered features)
    input_df_imputed_array = imputer.transform(X_eng_api_reordered)
    input_df_imputed = pd.DataFrame(input_df_imputed_array, columns=all_engineered_feature_names)

    # Select Top N features as used for the final model
    input_df_selected = input_df_imputed[selected_feature_names]

    # Make prediction
    prediction = model.predict(input_df_selected)

    return {"predicted_strength": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
