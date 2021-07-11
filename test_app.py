from fastapi.testclient import TestClient
from main import app


def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        assert response.status_code == 200
        assert response.json() == {"ping":"pong"}

def test_pred_virginica():
    payload = {
      "sepal_length": 3,
      "sepal_width": 5,
      "petal_length": 3.2,
      "petal_width": 4.4
    }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Virginica"}

def test_pred_Versicolour():
    payload = {
            "sepal_length": 5.7,
            "sepal_width": 2.9,
            "petal_length": 4.2,
            "petal_width": 1.3
        }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Versicolour"}

def test_pred_Setosa():
    payload = {
            "sepal_length": 4.6,
            "sepal_width": 3.1,
            "petal_length": 1.5,
            "petal_width": 0.2
        }
    with TestClient(app) as client:
        response = client.post('/predict_flower', json=payload)
        assert response.status_code == 200
        assert response.json() == {'flower_class': "Iris Setosa"}