import asyncio
import time
import httpx
from .config import API_KEY

async def test_async_predictions():
    """
    Test asynchronous behavior and background tasks by sending POST requests to /predict/{model_name}.
    Includes authentication.
    """
    # URLs pour les endpoints (lr pour logistic, rd pour random forest)
    url_lr = "http://127.0.0.1:8000/predict/lr"
    url_rf = "http://127.0.0.1:8000/predict/rd"
    data = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}  # Correspond à IrisData
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY  # Utilise x-api-key pour l'auth
    }
    
    async def make_request(url, request_id):
        start = time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            end = time.time()
            print(f"Request {request_id} to {url}: Status {response.status_code}, Time: {end - start:.2f}s")
            if response.status_code != 200:
                print(f"Error: {response.text}")
            return response.json() if response.status_code == 200 else None
    
    # 4 requêtes : 2 pour chaque modèle
    tasks = [
        make_request(url_lr, 1),
        make_request(url_lr, 2),
        make_request(url_rf, 3),
        make_request(url_rf, 4),
    ]
    
    start_total = time.time()
    results = await asyncio.gather(*tasks)
    end_total = time.time()
    
    print(f"\nTotal time for all requests: {end_total - start_total:.2f}s")
    print("Results:", [r for r in results if r])  # Ignore les None (erreurs)

if __name__ == "__main__":
    asyncio.run(test_async_predictions())