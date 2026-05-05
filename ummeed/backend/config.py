"""Settings loaded from environment / .env file."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql://ummeed:ummeed@localhost:5432/ummeed"

    JWT_SECRET: str = "dev-secret-change-me"
    JWT_ALGORITHM: str = "HS256"

    FAST2SMS_API_KEY: str = ""
    TWITTER_BEARER_TOKEN: str = ""

    PHOTO_STORAGE_PATH: str = "./storage/photos"

    OTP_TTL_SECONDS: int = 300            # 5 min for the OTP itself
    OTP_SESSION_TTL_SECONDS: int = 1800   # 30 min token to submit a sighting
    MODERATOR_TOKEN_TTL_HOURS: int = 24


settings = Settings()
