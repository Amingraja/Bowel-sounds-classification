from typing import List

from fastapi import APIRouter, Depends

from app.schemas.prediction_schemas import PredictionRequestSchema, PredictionResponseSchema
from app.api.dependencies.services import get_prediction_service

router = APIRouter()


@router.post("/bowel-sound/classify", response_model=List[PredictionResponseSchema])
async def classify_bowel_sound(request: PredictionRequestSchema,
                                  prediction_service=Depends(get_prediction_service)):
    prediction = prediction_service.predict(request)
    return prediction