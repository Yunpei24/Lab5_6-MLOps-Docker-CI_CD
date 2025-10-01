"""write a function to check if the key from API Key Header matches the one defined in .env"""
from fastapi import HTTPException, Security, Depends
from .config import API_KEY
from fastapi.security.api_key import APIKeyHeader

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# async def verify_api_key(x_api_key: str = Security(api_key_header)):
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid or missing API Key")
#     return True

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Usage example in a FastAPI endpoint:
# from fastapi import Depends
# @app.get("/secure-endpoint")
# async def secure_endpoint(api_key_valid: bool = Depends(verify_api_key)):
#     return {"message": "You have access to this secure endpoint"}
