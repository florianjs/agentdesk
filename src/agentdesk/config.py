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

    model_fallback: str = "google/gemini-2.5-flash"

    # Upstream resilience, shared by every provider call.
    retry_max_attempts: int = 3
    retry_base_delay_s: float = 1.0

    # Reasoning models spend hundreds of tokens before writing a word; a tight cap starves them
    # into returning nothing at all rather than into being brief.
    judge_max_tokens: int = 2000

    # Braintrust (optional) — versioned trajectory-eval scores.
    braintrust_api_key: str = ""
    braintrust_api_url: str = "https://api.braintrust.dev"
    braintrust_project_id: str = ""

    database_url: str = "postgresql+asyncpg://postgres:dev@localhost:5433/agentdesk"

    # The agent's tools are the two other services.
    triagely_url: str = "http://localhost:8000/v1"
    triagely_api_key: str = "tg_dev"
    docpilot_url: str = "http://localhost:8001/v1"
    docpilot_api_key: str = "dp_dev"
    docpilot_collection: str = "fastapi"

    # Budgets. Four of them, because each one stops a different runaway: a loop that never
    # concludes, a conversation that grows without bound, a bill, and a request that hangs.
    max_iterations: int = 10
    max_tokens_total: int = 50_000
    # USD: the currency the provider reports. Refund amounts are EUR — customer money and
    # provider spend are two different systems and mixing their units invites a wrong number.
    max_cost_usd: float = 0.50
    max_wall_seconds: float = 120.0


settings = Settings()
