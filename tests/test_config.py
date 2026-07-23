from agentic_secretary.config import _load_settings, settings


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
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "embedding_model_name")
    assert hasattr(settings, "chroma_api_key")
    assert hasattr(settings, "chroma_tenant")
    assert hasattr(settings, "chroma_database")
    assert hasattr(settings, "reranker_model_name")


def test_default_embedding_model_name_is_text_embedding_3_small():
    assert settings.embedding_model_name == "text-embedding-3-small"


def test_default_reranker_model_name_is_bge_reranker_v2_m3():
    assert settings.reranker_model_name == "BAAI/bge-reranker-v2-m3"


def test_embedding_model_name_reads_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "text-embedding-3-large")
    assert _load_settings().embedding_model_name == "text-embedding-3-large"


def test_reranker_model_name_reads_env_override(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
    assert _load_settings().reranker_model_name == "BAAI/bge-reranker-base"


def test_openai_and_chroma_settings_read_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("CHROMA_API_KEY", "chroma-test-key")
    monkeypatch.setenv("CHROMA_TENANT", "test-tenant")
    monkeypatch.setenv("CHROMA_DATABASE", "test-database")

    reloaded = _load_settings()

    assert reloaded.openai_api_key == "sk-test-openai"
    assert reloaded.chroma_api_key == "chroma-test-key"
    assert reloaded.chroma_tenant == "test-tenant"
    assert reloaded.chroma_database == "test-database"


def test_seed_token_path_defaults_and_is_isolated_from_runtime_token_path():
    assert settings.google_seed_token_path == "seed_token.json"
    assert settings.google_seed_token_path != settings.google_token_path


def test_cleanup_token_path_defaults_and_is_isolated_from_other_token_paths():
    assert settings.google_cleanup_token_path == "cleanup_token.json"
    assert settings.google_cleanup_token_path != settings.google_token_path
    assert settings.google_cleanup_token_path != settings.google_seed_token_path
