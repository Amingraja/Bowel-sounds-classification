from dataclasses import dataclass
from typing import Tuple


class ServerSetting:
    host: str = "0.0.0.0"
    port: int = 8000
    

@dataclass
class ModelInfo:
    PATH: str
    NAME: str


class ModelSetting:
    MODEL_INFOS: Tuple[ModelInfo] = (
        ModelInfo(
            PATH="models/fine_tuned_model_microsoft_wavlm-base",
            NAME="microsoft/wavlm-base"
        ),
        # ADD ADDITIONAL MODELS IF NEEDED
    )
        

class Settings:
    server: ServerSetting = ServerSetting()
    models: ModelSetting = ModelSetting()


settings = Settings()