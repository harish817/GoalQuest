from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GQ_",
        env_file=(".env", "goalquest/.env"),
        extra="ignore",
    )

    model: str = "openai/gpt-oss-120b"
    database_url: str = "sqlite:///./goalquest.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_path: str = "./.chroma"

    groq_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GQ_GROQ_API_KEY", "GROQ_API_KEY"),
    )
    hf_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GQ_HF_API_KEY", "HF_API_KEY"),
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GQ_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )

    # Game balance knobs (kept in config so they can be tuned without code changes)
    daily_xp_cap: int = 400
    streak_freeze_every_n_days: int = 7
    max_tasks_per_day: int = 3


settings = Settings()
