from agentic_secretary.config import settings


def test_default_model_name_is_haiku():
    assert settings.model_name == "claude-haiku-4-5"


def test_default_agent_model_name_is_sonnet():
    # The agent node orchestrates multi-step tool calls and narrates its
    # own actions back to the human -- the "harder-reasoning node" case
    # the spec allows a stronger model for, distinct from the classification/
    # extraction nodes that stay on the cheaper default.
    assert settings.agent_model_name == "claude-sonnet-4-5"
    assert settings.agent_model_name != settings.model_name


def test_settings_expose_required_fields():
    assert hasattr(settings, "anthropic_api_key")
    assert hasattr(settings, "langsmith_api_key")
    assert hasattr(settings, "langsmith_tracing")
    assert hasattr(settings, "langsmith_project")
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
