import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib # Added for saving artifacts
import json # Added for saving lists

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/ektanegi25/Cement-strength-prediction-project/main/cement_data.csv")

# Preprocessing: Simplify column names and strip spaces
original_columns_list = df.columns.to_list() # Keep original for API input reference
new_columns = [col.split("(")[0].strip() for col in original_columns_list]
df.columns = new_columns

# Preprocessing: Drop duplicates
df.drop_duplicates(keep='first', inplace=True)

# Preprocessing: Outlier capping
outliers_columns = ['Blast Furnace Slag', "Water", "Superplasticizer", 'Fine Aggregate', 'Age']
def outlier_capping(dataframe: pd.DataFrame, outliers_cols: list):
    df_copy = dataframe.copy()
    for i in outliers_cols:
        q1 = df_copy[i].quantile(0.25)
        q3 = df_copy[i].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        df_copy.loc[df_copy[i] > upper_limit, i] = upper_limit
        df_copy.loc[df_copy[i] < lower_limit, i] = lower_limit
    return df_copy
df = outlier_capping(dataframe=df, outliers_cols=outliers_columns)

# Feature Engineering
X_orig_df = df.drop('Concrete compressive strength', axis=1) # Renamed for clarity
y = df['Concrete compressive strength']
X_eng = X_orig_df.copy()
epsilon = 1e-6

X_eng['Water_to_Cement'] = X_orig_df['Water'] / (X_orig_df['Cement'] + epsilon)
X_eng['Superplasticizer_to_Cement'] = X_orig_df['Superplasticizer'] / (X_orig_df['Cement'] + epsilon)
X_eng['FlyAsh_to_Cement'] = X_orig_df['Fly Ash'] / (X_orig_df['Cement'] + epsilon)
X_eng['Slag_to_Cement'] = X_orig_df['Blast Furnace Slag'] / (X_orig_df['Cement'] + epsilon)
X_eng['FineAgg_to_CoarseAgg'] = X_orig_df['Fine Aggregate'] / (X_orig_df['Coarse Aggregate'] + epsilon)
X_eng['CoarseAgg_to_Cement'] = X_orig_df['Coarse Aggregate'] / (X_orig_df['Cement'] + epsilon)
X_eng['FineAgg_to_Cement'] = X_orig_df['Fine Aggregate'] / (X_orig_df['Cement'] + epsilon)

poly_features_cols = ['Cement', 'Water', 'Age']
poly_transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False) # Renamed
X_poly = poly_transformer.fit_transform(X_orig_df[poly_features_cols])
poly_feature_names = poly_transformer.get_feature_names_out(poly_features_cols)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X_orig_df.index)
X_eng = pd.concat([X_eng, X_poly_df.drop(columns=poly_features_cols)], axis=1)

X = X_eng
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Save all engineered feature names before imputation
all_engineered_feature_names = xtrain.columns.tolist()

knn_imputer = KNNImputer(n_neighbors=3) # Renamed
xtrain_imputed = knn_imputer.fit_transform(xtrain) # Fit on xtrain (all engineered features)
xtest_imputed = knn_imputer.transform(xtest)
xtrain_imputed_df = pd.DataFrame(xtrain_imputed, columns=xtrain.columns)
xtest_imputed_df = pd.DataFrame(xtest_imputed, columns=xtest.columns)

# GBDT for importance (using parameters from previous best run on all eng features)
gbdt_params_full_features = {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 6, 'n_estimators': 300}
gbdt_for_importance = GradientBoostingRegressor(random_state=42, **gbdt_params_full_features)
gbdt_for_importance.fit(xtrain_imputed_df, ytrain)
importances = gbdt_for_importance.feature_importances_
feature_importance_df = pd.DataFrame({'feature': all_engineered_feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
print("\n========== Feature Importances (New Features - GBDT All Eng. Features) ==========")
print(feature_importance_df.head(20))
plt.figure(figsize=(10, 12))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
plt.title('Top 20 Feature Importances from GBDT (All Eng. Features)')
plt.tight_layout()
plt.savefig("feature_importances_new.png")

top_n = 15
selected_feature_names = feature_importance_df['feature'].head(top_n).tolist() # Renamed
print(f"\nSelected Top {top_n} Features: {selected_feature_names}")

xtrain_selected = xtrain_imputed_df[selected_feature_names]
xtest_selected = xtest_imputed_df[selected_feature_names]

# --- GBDT with selected features (using params from full-feature GBDT tuning) ---
print(f"\n========== GBDT Performance with Top {top_n} Selected Features (Params from full-feature GBDT) ==========")
gb_rg_selected_baseline = GradientBoostingRegressor(random_state=42, **gbdt_params_full_features) # Renamed
gb_rg_selected_baseline.fit(xtrain_selected, ytrain)
y_pred_gb_selected = gb_rg_selected_baseline.predict(xtest_selected)
r2_gb_selected = r2_score(ytest, y_pred_gb_selected)
print(f"R2 Score: {r2_gb_selected:.4f}")
# ... (other print statements for GBDT MSE, MAE)

# --- SVR with selected features ---
print(f"\n========== SVR Performance with Top {top_n} Selected Features ==========")
scaler_svr = StandardScaler()
xtrain_selected_scaled_svr = scaler_svr.fit_transform(xtrain_selected)
xtest_selected_scaled_svr = scaler_svr.transform(xtest_selected)
svr_model = SVR()
svr_model.fit(xtrain_selected_scaled_svr, ytrain)
y_pred_svr = svr_model.predict(xtest_selected_scaled_svr)
r2_svr = r2_score(ytest, y_pred_svr)
print(f"SVR R2 Score: {r2_svr:.4f}")
# ... (other print statements for SVR MSE, MAE)

# --- XGBoost with selected features ---
print(f"\n========== XGBoost Regressor Performance with Top {top_n} Selected Features ==========")
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(xtrain_selected, ytrain)
y_pred_xgb = xgb_model.predict(xtest_selected)
r2_xgb = r2_score(ytest, y_pred_xgb)
print(f"XGB R2 Score: {r2_xgb:.4f}")
# ... (other print statements for XGB MSE, MAE)

# Determine best model
models_performance = {
    "GBDT_Selected_Baseline": (r2_gb_selected, gb_rg_selected_baseline, xtrain_selected, xtest_selected, False),
    "SVR_Selected_Scaled": (r2_svr, SVR(), xtrain_selected_scaled_svr, xtest_selected_scaled_svr, True), # Use new SVR instance for tuning
    "XGB_Selected": (r2_xgb, XGBRegressor(random_state=42), xtrain_selected, xtest_selected, False) # Use new XGB instance
}
best_model_name = max(models_performance, key=lambda k: models_performance[k][0])
_, best_model_initial_obj, best_model_train_data, best_model_test_data, best_model_requires_scaling = models_performance[best_model_name]
print(f"\n--- Best performing model before extensive tuning: {best_model_name} with R2: {models_performance[best_model_name][0]:.4f} ---")

# Cross-Validation
print(f"\n========== 5-Fold Cross-Validation for {best_model_name} ==========")
# Re-initialize model for CV based on its type
if "SVR" in best_model_name:
    cv_model_obj = SVR() # Default SVR for CV if SVR is best
    # CV data should be scaled if the model requires it
    # scaler_for_cv = StandardScaler()
    # train_data_for_cv_scaled = scaler_for_cv.fit_transform(xtrain_imputed_df[selected_feature_names]) # Scale selected features from original imputed df
    # cv_train_data_final = train_data_for_cv_scaled
    # Using already scaled data for SVR to avoid refitting scaler in CV for simplicity here.
    cv_train_data_final = best_model_train_data
elif "XGB" in best_model_name:
    cv_model_obj = XGBRegressor(random_state=42)
    cv_train_data_final = xtrain_selected # Use unscaled selected features
else: # GBDT
    cv_model_obj = GradientBoostingRegressor(random_state=42, **gbdt_params_full_features)
    cv_train_data_final = xtrain_selected # Use unscaled selected features

cv_scores = cross_val_score(cv_model_obj, cv_train_data_final, ytrain, cv=5, scoring='r2')
print(f"Mean R2 Score: {np.mean(cv_scores):.4f}, Std Dev: {np.std(cv_scores):.4f}")

# Hyperparameter Tuning for the identified best model
print(f"\n========== Hyperparameter Tuning (RandomizedSearchCV) for {best_model_name} ==========")
param_dist = {}
model_to_tune = None
tuning_train_data = best_model_train_data # Use data appropriate for the model (scaled/unscaled)
tuning_test_data = best_model_test_data

if "GBDT" in best_model_name:
    param_dist = {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.15],
                  'max_depth': [3, 4, 5, 6, 7], 'min_samples_split': [2, 4, 6, 8],
                  'min_samples_leaf': [1, 2, 3, 4], 'subsample': [0.7, 0.8, 0.9, 1.0]}
    model_to_tune = GradientBoostingRegressor(random_state=42)
elif "SVR" in best_model_name:
    param_dist = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1], 'kernel': ['rbf']} # Simplified SVR grid
    model_to_tune = SVR()
else: # XGB
    param_dist = {'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.15],
                  'max_depth': [3, 4, 5, 6, 7], 'subsample': [0.7, 0.8, 0.9, 1.0],
                  'colsample_bytree': [0.7, 0.8, 0.9, 1.0]}
    model_to_tune = XGBRegressor(random_state=42)

random_search = RandomizedSearchCV(model_to_tune, param_distributions=param_dist, n_iter=10, # Reduced n_iter further
                                   cv=3, scoring='r2', random_state=42, verbose=0, n_jobs=-1)
random_search.fit(tuning_train_data, ytrain)

print(f"Best Parameters from RandomizedSearch: {random_search.best_params_}")
final_trained_model = random_search.best_estimator_ # This is the model to save

y_pred_tuned = final_trained_model.predict(tuning_test_data)
r2_tuned = r2_score(ytest, y_pred_tuned)
print(f"\n========== {best_model_name} Performance after RandomizedSearchCV ==========")
print(f"R2 Score: {r2_tuned:.4f}")
# ... (other print statements for tuned MSE, MAE)

# Save artifacts
print("\n--- Saving artifacts ---")
joblib.dump(final_trained_model, 'final_model.joblib')
joblib.dump(knn_imputer, 'knn_imputer.joblib')
joblib.dump(poly_transformer, 'poly_transformer.joblib')

artifacts_to_save = {
    'original_input_columns': new_columns, # Original 8 features for API input
    'poly_features_cols': poly_features_cols,
    'all_engineered_feature_names': all_engineered_feature_names, # For reconstructing full feature set before imputation
    'selected_feature_names': selected_feature_names # Top N features for final model
}
with open('feature_lists.json', 'w') as f:
    json.dump(artifacts_to_save, f)

# If SVR was chosen and scaled, save its scaler too.
if best_model_name == "SVR_Selected_Scaled":
     joblib.dump(scaler_svr, 'svr_scaler.joblib')


print("\nconcrete_strength_predictor.py script with artifact saving finished.")
