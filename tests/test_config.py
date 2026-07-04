from agentic_secretary.config import settings


def test_default_model_name_is_haiku():
    assert settings.model_name == "claude-haiku-4-5"


def test_settings_expose_required_fields():
    assert hasattr(settings, "anthropic_api_key")
    assert hasattr(settings, "langsmith_api_key")
    assert hasattr(settings, "google_client_secret_path")
    assert hasattr(settings, "google_token_path")
