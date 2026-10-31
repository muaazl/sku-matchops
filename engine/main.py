"""
SKU MatchOps ML Engine - Microservice Application Entry Point
"""

if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("ENGINE_PORT", 8001))
    host = os.getenv("ENGINE_HOST", "0.0.0.0")
    uvicorn.run("engine.main:app", host=host, port=port, reload=False)
