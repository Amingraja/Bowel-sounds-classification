from fastapi import FastAPI

from .routes.home_route import router as home_router
from .routes.prediction_route import router as prediction_route

app = FastAPI()

app.include_router(home_router)
app.include_router(prediction_route)
