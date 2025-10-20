"""
Script to run the Vector Database API server.

This script starts the FastAPI application using uvicorn.
"""

import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))

    logger.info(f"Starting Vector Database API on {host}:{port}")
    logger.info(f"Workers: {workers}, Reload: {reload}")
    logger.info("API Documentation available at /docs")

    uvicorn.run(
        "app.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Reload doesn't work with multiple workers
        log_level="info",
    )
