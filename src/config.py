import os

CODEXIFY_API_SERVER_LISTEN_PORT = int(os.environ.get("CODEXIFY_API_SERVER_LISTEN_PORT", 8089))
UVICORN_NUMBER_OF_WORKERS = int(os.environ.get("UVICORN_NUMBER_OF_WORKERS", 3))

# Add these configurations
REDIS_CONFIG = {
    'max_memory': '4gb',  # Adjust based on your server capacity
    'max_memory_policy': 'allkeys-lru'
}

option = {
    "host": "0.0.0.0",
    "port": CODEXIFY_API_SERVER_LISTEN_PORT,
    "workers": UVICORN_NUMBER_OF_WORKERS
}
