import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

root_dir = os.path.dirname(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


@pytest.fixture(scope="session", autouse=True)
def mock_config():
    config_mock = MagicMock()
    config_mock.API_KEY = "mysecretkey123"
    config_mock.LOGISTIC_MODEL = "/app/models/logistic_regression.pkl"
    config_mock.RF_MODEL = "/app/models/random_forest.pkl"
    config_mock.MODELS_DIR = "/app/models"
    
    with patch.dict('sys.modules', {'app.config': config_mock}):
        yield config_mock


@pytest.fixture(autouse=True)
def mock_joblib():
    """Mock joblib.load pour éviter de charger de vrais modèles"""
    with patch('joblib.load') as mock_load:
        # Créer des modèles mockés
        mock_lr_model = Mock()
        mock_rf_model = Mock()
        
        # Configurer les prédictions mockées
        mock_lr_model.predict.return_value = [0]  # Setosa
        mock_rf_model.predict.return_value = [1]  # Versicolor
        
        def load_side_effect(*args, **kwargs):
            path = str(args[0]) if args else ""
            if 'logistic' in path:
                return mock_lr_model
            else:
                return mock_rf_model
        
        mock_load.side_effect = load_side_effect
        yield mock_load


@pytest.fixture
def client(mock_config, mock_joblib):
    """Create a test client that handles the lifespan of the application."""
    with TestClient(app) as test_client:
        yield test_client


def test_read_main(client):
    """Test l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_health_check(client):
    """Test le health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_available_models(client):
    """Test l'endpoint des modèles disponibles"""
    headers = {"x-api-key": "mysecretkey123"}
    response = client.get("/models", headers=headers)
    assert response.status_code == 200
    assert "available_models" in response.json()
    assert isinstance(response.json()["available_models"], list)


def test_predict_invalid_model_name(client):
    """Test prédiction avec nom de modèle invalide"""
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    headers = {"x-api-key": "mysecretkey123"}
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
    headers = {"x-api-key": "mysecretkey123"}
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
    headers = {"x-api-key": "mysecretkey123"}
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
    headers = {"x-api-key": "mysecretkey123"}
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
    headers = {"x-api-key": "mysecretkey123"}
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
    headers = {"x-api-key": "mysecretkey123"}
    response = client.post("/predict/lr", json=data, headers=headers)
    
    assert response.status_code == 422  # Erreur de validation