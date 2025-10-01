from pydantic import BaseModel, Field


class PredictionRequestSchema(BaseModel):
    audio_path: str = Field(
        description="The path of audio file to predict"
    )


class PredictionResponseSchema(BaseModel):
    start_time: float
    end_time: float
    label: str