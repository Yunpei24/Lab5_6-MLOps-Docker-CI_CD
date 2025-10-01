#!/bin/bash
# Test async pour logistic_reg
time curl -X POST "http://127.0.0.1:8000/predict/logistic_reg" -H "Content-Type: application/json" -d '{"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' &
time curl -X POST "http://127.0.0.1:8000/predict/logistic_reg" -H "Content-Type: application/json" -d '{"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' &

# Test async pour random_forest
time curl -X POST "http://127.0.0.1:8000/predict/random_forest" -H "Content-Type: application/json" -d '{"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' &
time curl -X POST "http://127.0.0.1:8000/predict/random_forest" -H "Content-Type: application/json" -d '{"data": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' &

wait  # Attend que toutes les requêtes finissent