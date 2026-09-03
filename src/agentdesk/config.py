"""Configuration — read from the environment and `.env`, validated at startup.

Invalid config crashes startup, not the first request. That is intentional.
In production only real environment variables apply; a missing `.env` is ignored.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""

    model_smart: str = "anthropic/claude-sonnet-4.5"
    model_cheap: str = "anthropic/claude-haiku-4.5"
    # The eval judge must be a STRONGER model than the one being judged.
    model_judge: str = "anthropic/claude-sonnet-4.5"

    # Braintrust (optional) — versioned trajectory-eval scores.
    braintrust_api_key: str = ""
    braintrust_api_url: str = "https://api.braintrust.dev"
    braintrust_project_id: str = ""

    database_url: str = "postgresql+asyncpg://postgres:dev@localhost:5433/agentdesk"

    # The agent's tools are the two other services.
    triagely_url: str = "http://localhost:8000/v1"
    docpilot_url: str = "http://localhost:8001/v1"
    suite_api_key: str = "tg_dev"


settings = Settings()
