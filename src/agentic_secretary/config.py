import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = "claude-haiku-4-5"


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    langsmith_api_key: str | None
    google_client_secret_path: str
    google_token_path: str
    model_name: str


def _load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        google_client_secret_path=os.getenv(
            "GOOGLE_CLIENT_SECRET_PATH", "credentials.json"
        ),
        google_token_path=os.getenv("GOOGLE_TOKEN_PATH", "token.json"),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
    )


settings = _load_settings()
