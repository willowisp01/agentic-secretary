from agentic_secretary.config import settings


def test_default_model_name_is_haiku():
    assert settings.model_name == "claude-haiku-4-5"


def test_settings_expose_required_fields():
    assert hasattr(settings, "anthropic_api_key")
    assert hasattr(settings, "langsmith_api_key")
    assert hasattr(settings, "google_client_secret_path")
    assert hasattr(settings, "google_token_path")
    assert hasattr(settings, "google_seed_token_path")
    assert hasattr(settings, "google_cleanup_token_path")


def test_seed_token_path_defaults_and_is_isolated_from_runtime_token_path():
    assert settings.google_seed_token_path == "seed_token.json"
    assert settings.google_seed_token_path != settings.google_token_path


def test_cleanup_token_path_defaults_and_is_isolated_from_other_token_paths():
    assert settings.google_cleanup_token_path == "cleanup_token.json"
    assert settings.google_cleanup_token_path != settings.google_token_path
    assert settings.google_cleanup_token_path != settings.google_seed_token_path
