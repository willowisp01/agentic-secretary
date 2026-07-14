import os
from dataclasses import dataclass
from datetime import timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = "claude-haiku-4-5"

# The seeded burner account's fictional persona is anchored to this
# timezone. Bare (offset-less) clock times -- "+1d 09:00" in a seed fixture,
# "9:15am" mentioned in an email body -- are resolved against this rather
# than UTC, so "9am" actually means 9am in the persona's local time instead
# of 9am UTC (which would display as 5pm locally).
DEMO_TIMEZONE = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None
    langsmith_api_key: str | None
    google_client_secret_path: str
    google_token_path: str
    google_seed_token_path: str
    google_cleanup_token_path: str
    model_name: str


def _load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        google_client_secret_path=os.getenv(
            "GOOGLE_CLIENT_SECRET_PATH", "credentials.json"
        ),
        google_token_path=os.getenv("GOOGLE_TOKEN_PATH", "token.json"),
        # Deliberately a separate cache from google_token_path: seeding needs
        # broader (write) scopes than the agent's runtime grant, and mixing
        # the two token files would leak that elevated grant into runtime use.
        google_seed_token_path=os.getenv("GOOGLE_SEED_TOKEN_PATH", "seed_token.json"),
        # Separate again from both of the above: gmail.modify (needed to trash
        # arbitrary messages) is a strictly more dangerous grant than insert-only
        # seeding, so it gets its own cache too rather than expanding seed_token.json.
        google_cleanup_token_path=os.getenv(
            "GOOGLE_CLEANUP_TOKEN_PATH", "cleanup_token.json"
        ),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
    )


settings = _load_settings()
