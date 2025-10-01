import pytest
import sys
import os
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import API_KEY



@pytest.fixture
def mock_models(mocker):
    mock_dict = {"logistic_model": MagicMock, "rf_model": MagicMock}
    m = mocker.patch(
        "app.main.models",
        return_value=mock_dict,
    )
    m.keys.return_value = mock_dict.keys()
    return m


@pytest.fixture
def client(mock_models):
    """Create a test client that handles the lifespan of the application."""
    with TestClient(app) as client:
        yield client


def test_read_main(client):
    """Test l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_available_models():
    with TestClient(app) as client:
        headers = {"x-api-key": API_KEY}
        response = client.get("/models", headers=headers)
        assert response.status_code == 200
        assert isinstance(response.json()["available_models"], list)


def test_predict_invalid_model_name(client):
    """Test prédiction avec nom de modèle invalide"""
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/predict/invalid_model", json=data, headers=headers)
    
    assert response.status_code == 400
    assert "Invalid model name" in response.json()["detail"]


def test_predict_valid_lr_model(client):
    """Test prédiction avec modèle logistic regression valide"""

    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/predict/lr", json=data, headers=headers)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction label" in response_data
    assert "name" in response_data
    assert isinstance(response_data["prediction label"], int)
    assert response_data["name"] in ["Setosa", "Versicolor", "Virginica"]


def test_predict_valid_rd_model(client):
    data = {
        "sepal_length": 6.2,
        "sepal_width": 3.4,
        "petal_length": 5.4,
        "petal_width": 2.3
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/predict/rd", json=data, headers=headers)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction label" in response_data
    assert "name" in response_data


def test_predict_missing_api_key(client):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict/lr", json=data)
    
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_predict_invalid_api_key(client):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": "wrongkey"}
    response = client.post("/predict/lr", json=data, headers=headers)
    
    assert response.status_code == 401
    assert "Invalid API key" in response.json()["detail"]


def test_predict_logistic_regression_endpoint(client):
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/prediction/logistic_reg", json=data, headers=headers)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction label" in response_data
    assert "name" in response_data


def test_predict_random_forest_endpoint(client):
    data = {
        "sepal_length": 6.2,
        "sepal_width": 3.4,
        "petal_length": 5.4,
        "petal_width": 2.3
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/prediction/random_forest", json=data, headers=headers)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction label" in response_data
    assert "name" in response_data


def test_models_endpoint_missing_api_key(client):
    response = client.get("/models")
    assert response.status_code == 401


def test_predict_invalid_input_data(client):
    data = {
        "sepal_length": -1,  # Valeur négative invalide
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": API_KEY}
    response = client.post("/predict/lr", json=data, headers=headers)
    
    assert response.status_code == 422  # Erreur de validation