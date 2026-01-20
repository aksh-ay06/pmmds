"""Application settings using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration.

    Loads from environment variables with PMMDS_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PMMDS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_title: str = "PMMDS Inference API"
    api_version: str = "0.1.0"

    # Database settings
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "pmmds"
    db_password: str = "pmmds"
    db_name: str = "pmmds"

    # Model settings
    model_name: str = "baseline"
    model_version: str = "latest"

    # Logging
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_sync(self) -> str:
        """Construct sync database URL for migrations."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
