import pytest
from fastapi.testclient import TestClient
# Assuming main.py and its app object are in the same directory or accessible
# If main.py is in the root and tests are in tests/, you might need to adjust path
# For this task structure, assume they are at the same level for simplicity of import
from main import app

client = TestClient(app)

def test_predict_endpoint_valid_input():
    payload = {
        "Cement": 300.0,
        "Blast_Furnace_Slag": 100.0,
        "Fly_Ash": 50.0,
        "Water": 180.0,
        "Superplasticizer": 5.0,
        "Coarse_Aggregate": 950.0,
        "Fine_Aggregate": 750.0,
        "Age": 28.0
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "predicted_strength" in json_response
    assert isinstance(json_response["predicted_strength"], float)
    # Check if prediction is within a very broad reasonable range
    assert 0 < json_response["predicted_strength"] < 150

def test_predict_endpoint_invalid_input_missing_field():
    payload = {
        "Cement": 300.0,
        # "Blast_Furnace_Slag": 100.0, # Missing field
        "Fly_Ash": 50.0,
        "Water": 180.0,
        "Superplasticizer": 5.0,
        "Coarse_Aggregate": 950.0,
        "Fine_Aggregate": 750.0,
        "Age": 28.0
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422  # Unprocessable Entity for Pydantic validation error

def test_predict_endpoint_invalid_input_wrong_type():
    payload = {
        "Cement": "not-a-float", # Incorrect data type
        "Blast_Furnace_Slag": 100.0,
        "Fly_Ash": 50.0,
        "Water": 180.0,
        "Superplasticizer": 5.0,
        "Coarse_Aggregate": 950.0,
        "Fine_Aggregate": 750.0,
        "Age": 28.0
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422

def test_read_main_root():
    response = client.get("/") # Assuming you might add a root endpoint later; if not, this will fail or can be removed
    assert response.status_code == 404 # No root endpoint defined in main.py yet
