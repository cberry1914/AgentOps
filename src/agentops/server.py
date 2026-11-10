
"""
AgentOps - API Server Entry Point

Starts the FastAPI application using Uvicorn.
"""

import os

import uvicorn


def main():
    """Start the AgentOps API server."""

    host = os.getenv("AGENTOPS_HOST", "127.0.0.1")
    port = int(os.getenv("AGENTOPS_PORT", "8000"))

    print("Starting AgentOps API...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"API documentation: http://{host}:{port}/docs")

    uvicorn.run(
        "agentops.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
