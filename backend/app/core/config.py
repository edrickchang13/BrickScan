from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REBRICKABLE_API_KEY: str
    BRICKLINK_CONSUMER_KEY: str
    BRICKLINK_CONSUMER_SECRET: str
    BRICKLINK_TOKEN: str
    BRICKLINK_TOKEN_SECRET: str
    GEMINI_API_KEY: str
    ML_MODEL_PATH: str = "/app/models/lego_classifier.onnx"
    # "dinov2" or "efficientnet" — auto-detected from export_info.json if omitted
    ML_MODEL_TYPE: str = ""
    CONFIDENCE_THRESHOLD: float = 0.75
    S3_BUCKET: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
