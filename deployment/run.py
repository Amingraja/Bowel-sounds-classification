if __name__ == "__main__":
    from app.api.main import app
    from app.core.config import Settings
    import uvicorn

    uvicorn.run(app, host=Settings.server.host, port=Settings.server.port)
