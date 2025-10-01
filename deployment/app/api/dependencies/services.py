from typing_extensions import Annotated

from fastapi import Depends

from app.core.config import settings
from app.services.prediction_service import PredictionService


def get_prediction_service():
    return PredictionService(model_infos=settings.models.MODEL_INFOS)


PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]