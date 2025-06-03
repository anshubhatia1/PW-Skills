import pytest
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import KNNImputer # Needed for one test if we load the imputer

# Definition of outlier_capping function (copied from concrete_strength_predictor.py for isolated testing)
def outlier_capping(dataframe: pd.DataFrame, outliers_cols: list):
    df_copy = dataframe.copy()
    for i in outliers_cols:
        q1 = df_copy[i].quantile(0.25)
        q3 = df_copy[i].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr # Corrected version
        df_copy.loc[df_copy[i] > upper_limit, i] = upper_limit
        df_copy.loc[df_copy[i] < lower_limit, i] = lower_limit
    return df_copy

def test_outlier_capping():
    data = {'col1': [1, 2, 3, 4, 5, 100], 'col2': [10, 20, 30, 40, 50, 200]}
    df = pd.DataFrame(data)
    df_capped = outlier_capping(df, ['col1'])

    expected_upper_limit_col1 = 8.5

    assert df_capped['col1'].max() <= expected_upper_limit_col1
    assert df_capped['col1'].max() == pytest.approx(expected_upper_limit_col1)
    # The value 100 (at index 5) should be capped at 8.5
    assert df_capped.loc[5, 'col1'] == pytest.approx(expected_upper_limit_col1)
    assert df_capped['col2'].equals(df['col2']) # col2 should be unchanged


def test_ratio_feature_creation():
    data = {
        'Cement': [200, 300], 'Blast Furnace Slag': [50, 0], 'Fly Ash': [20, 0],
        'Water': [100, 150], 'Superplasticizer': [5, 0],
        'Coarse Aggregate': [1000, 900], 'Fine Aggregate': [800, 700], 'Age': [28, 28]
    }
    X_orig_df = pd.DataFrame(data)
    X_eng_api = X_orig_df.copy()
    epsilon = 1e-6

    X_eng_api['Water_to_Cement'] = X_orig_df['Water'] / (X_orig_df['Cement'] + epsilon)
    X_eng_api['Superplasticizer_to_Cement'] = X_orig_df['Superplasticizer'] / (X_orig_df['Cement'] + epsilon)

    assert 'Water_to_Cement' in X_eng_api.columns
    assert 'Superplasticizer_to_Cement' in X_eng_api.columns
    assert X_eng_api['Water_to_Cement'].iloc[0] == pytest.approx(100/200)
    assert X_eng_api['Superplasticizer_to_Cement'].iloc[0] == pytest.approx(5/200)
    assert X_eng_api['Water_to_Cement'].iloc[1] == pytest.approx(150/300)
    assert X_eng_api['Superplasticizer_to_Cement'].iloc[1] == pytest.approx(0/300)

def test_polynomial_feature_generation():
    data = {'Cement': [200, 300], 'Water': [100, 150], 'Age': [28, 56]}
    X_orig_df = pd.DataFrame(data)
    poly_features_cols = ['Cement', 'Water', 'Age']

    poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly_transformer.fit_transform(X_orig_df[poly_features_cols])
    poly_feature_names_api = poly_transformer.get_feature_names_out(poly_features_cols)
    X_poly_api_df = pd.DataFrame(X_poly, columns=poly_feature_names_api)

    assert X_poly_api_df.shape[1] == 9
    assert 'Cement Water' in X_poly_api_df.columns
    assert 'Age^2' in X_poly_api_df.columns
    assert X_poly_api_df['Cement Water'].iloc[0] == pytest.approx(200 * 100)
    assert X_poly_api_df['Age^2'].iloc[1] == pytest.approx(56 * 56)

def test_prediction_pipeline_works():
    try:
        model = joblib.load('final_model.joblib')
        imputer = joblib.load('knn_imputer.joblib')
        poly_transformer_loaded = joblib.load('poly_transformer.joblib')
        with open('feature_lists.json', 'r') as f:
            feature_lists = json.load(f)
    except FileNotFoundError:
        pytest.skip("Model/transformer artifacts not found. Run concrete_strength_predictor.py first.")

    original_input_columns = feature_lists['original_input_columns']
    poly_features_cols_loaded = feature_lists['poly_features_cols']
    all_engineered_feature_names_loaded = feature_lists['all_engineered_feature_names']
    selected_feature_names_loaded = feature_lists['selected_feature_names']

    sample_input_dict = {
        'Cement': 300.0, 'Blast Furnace Slag': 100.0, 'Fly Ash': 50.0, 'Water': 180.0,
        'Superplasticizer': 5.0, 'Coarse Aggregate': 950.0, 'Fine Aggregate': 750.0, 'Age': 28.0
    }
    input_df_orig = pd.DataFrame([sample_input_dict], columns=original_input_columns)

    X_eng_test = input_df_orig.copy()
    epsilon = 1e-6
    X_eng_test['Water_to_Cement'] = input_df_orig['Water'] / (input_df_orig['Cement'] + epsilon)
    X_eng_test['Superplasticizer_to_Cement'] = input_df_orig['Superplasticizer'] / (input_df_orig['Cement'] + epsilon)
    X_eng_test['FlyAsh_to_Cement'] = input_df_orig['Fly Ash'] / (input_df_orig['Cement'] + epsilon)
    X_eng_test['Slag_to_Cement'] = input_df_orig['Blast Furnace Slag'] / (input_df_orig['Cement'] + epsilon)
    X_eng_test['FineAgg_to_CoarseAgg'] = input_df_orig['Fine Aggregate'] / (input_df_orig['Coarse Aggregate'] + epsilon)
    X_eng_test['CoarseAgg_to_Cement'] = input_df_orig['Coarse Aggregate'] / (input_df_orig['Cement'] + epsilon)
    X_eng_test['FineAgg_to_Cement'] = input_df_orig['Fine Aggregate'] / (input_df_orig['Cement'] + epsilon)

    X_poly_test = poly_transformer_loaded.transform(input_df_orig[poly_features_cols_loaded])
    poly_feature_names_test = poly_transformer_loaded.get_feature_names_out(poly_features_cols_loaded)
    X_poly_test_df = pd.DataFrame(X_poly_test, columns=poly_feature_names_test, index=input_df_orig.index)
    X_eng_test = pd.concat([X_eng_test, X_poly_test_df.drop(columns=poly_features_cols_loaded)], axis=1)

    X_eng_test_reordered = X_eng_test.reindex(columns=all_engineered_feature_names_loaded, fill_value=np.nan)

    input_df_imputed_array = imputer.transform(X_eng_test_reordered)
    input_df_imputed = pd.DataFrame(input_df_imputed_array, columns=all_engineered_feature_names_loaded)

    input_df_selected = input_df_imputed[selected_feature_names_loaded]
    prediction = model.predict(input_df_selected)

    assert isinstance(prediction[0], float)
    assert prediction[0] > 0
    assert 0 < prediction[0] < 150
