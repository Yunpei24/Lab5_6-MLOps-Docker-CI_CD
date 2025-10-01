import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
def load_secret(secret_name: str, default: str = None) -> str:
    """Load secret from file or environment variable."""
    # Fallback to environment variable
    env_var = os.getenv(secret_name.upper(), default)
    if env_var:
        return env_var

    # Try to read from Docker secret file first
    secret_file = Path(f"/run/secrets/{secret_name}")
    if secret_file.exists():
        return secret_file.read_text().strip()
    
    # Final fallback to local secret file (for development)
    local_secret_file = Path(f"secrets/{secret_name}.txt")
    if local_secret_file.exists():
        return local_secret_file.read_text().strip()
    
    raise ValueError(f"Secret {secret_name} not found")

# Configuration from environment variables
LOGISTIC_MODEL = os.getenv("LOGISTIC_MODEL", "/app/models/logistic_regression.pkl")
RF_MODEL = os.getenv("RF_MODEL", "/app/models/random_forest.pkl")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")

API_KEY = load_secret("API_KEY")

 