from unittest.mock import MagicMock, patch

from google.auth.exceptions import RefreshError

from agentic_secretary.auth import get_credentials, load_credentials


def _fake_creds(valid=True, expired=False, refresh_token="refresh-token"):
    creds = MagicMock()
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = '{"token": "fake"}'
    return creds


def test_runs_consent_flow_when_no_cached_token(tmp_path):
    token_path = tmp_path / "token.json"
    new_creds = _fake_creds()

    with (
        patch("agentic_secretary.auth.settings") as mock_settings,
        patch("agentic_secretary.auth.InstalledAppFlow") as mock_flow_cls,
    ):
        mock_settings.google_token_path = str(token_path)
        mock_settings.google_client_secret_path = "credentials.json"
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow

        creds = get_credentials()

    mock_flow_cls.from_client_secrets_file.assert_called_once()
    mock_flow.run_local_server.assert_called_once()
    assert creds is new_creds
    assert token_path.read_text() == '{"token": "fake"}'


def test_reuses_valid_cached_token(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "cached"}')
    cached_creds = _fake_creds(valid=True)

    with (
        patch("agentic_secretary.auth.settings") as mock_settings,
        patch("agentic_secretary.auth.Credentials") as mock_creds_cls,
        patch("agentic_secretary.auth.InstalledAppFlow") as mock_flow_cls,
    ):
        mock_settings.google_token_path = str(token_path)
        mock_creds_cls.from_authorized_user_file.return_value = cached_creds

        creds = get_credentials()

    mock_flow_cls.from_client_secrets_file.assert_not_called()
    assert creds is cached_creds


def test_refreshes_expired_token_without_new_consent(tmp_path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "cached"}')
    expired_creds = _fake_creds(valid=False, expired=True)

    def _mark_valid_after_refresh(request):
        # Mirrors real Credentials.valid: a computed property ("has a token
        # and isn't expired") that a genuine successful refresh() flips to
        # True as a side effect -- this MagicMock needs to fake that too.
        expired_creds.valid = True

    expired_creds.refresh.side_effect = _mark_valid_after_refresh

    with (
        patch("agentic_secretary.auth.settings") as mock_settings,
        patch("agentic_secretary.auth.Credentials") as mock_creds_cls,
        patch("agentic_secretary.auth.InstalledAppFlow") as mock_flow_cls,
        patch("agentic_secretary.auth.Request"),
    ):
        mock_settings.google_token_path = str(token_path)
        mock_creds_cls.from_authorized_user_file.return_value = expired_creds

        creds = get_credentials()

    expired_creds.refresh.assert_called_once()
    mock_flow_cls.from_client_secrets_file.assert_not_called()
    assert creds is expired_creds
    assert token_path.read_text() == '{"token": "fake"}'


def test_falls_back_to_consent_flow_when_refresh_token_invalid(tmp_path):
    # Reproduces a live crash: a cached refresh token can itself be
    # revoked/expired (e.g. Google expires refresh tokens after 7 days for
    # OAuth clients in "Testing" publishing status), in which case
    # creds.refresh() raises RefreshError rather than returning refreshed
    # creds. That should fall back to a fresh consent flow, not crash.
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "cached"}')
    expired_creds = _fake_creds(valid=False, expired=True)
    expired_creds.refresh.side_effect = RefreshError(
        "invalid_grant: Token has been expired or revoked."
    )
    new_creds = _fake_creds()

    with (
        patch("agentic_secretary.auth.settings") as mock_settings,
        patch("agentic_secretary.auth.Credentials") as mock_creds_cls,
        patch("agentic_secretary.auth.InstalledAppFlow") as mock_flow_cls,
        patch("agentic_secretary.auth.Request"),
    ):
        mock_settings.google_token_path = str(token_path)
        mock_creds_cls.from_authorized_user_file.return_value = expired_creds
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow

        creds = get_credentials()

    expired_creds.refresh.assert_called_once()
    mock_flow_cls.from_client_secrets_file.assert_called_once()
    mock_flow.run_local_server.assert_called_once()
    assert creds is new_creds
    assert token_path.read_text() == '{"token": "fake"}'


def test_load_credentials_uses_given_scopes_and_token_path_not_runtime_settings():
    token_path_str = "some/other/seed_token.json"
    seed_scopes = ["https://www.googleapis.com/auth/gmail.insert"]
    new_creds = _fake_creds()

    with (
        patch("agentic_secretary.auth.Path") as mock_path_cls,
        patch("agentic_secretary.auth.InstalledAppFlow") as mock_flow_cls,
    ):
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_path_cls.return_value = mock_path
        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = new_creds
        mock_flow_cls.from_client_secrets_file.return_value = mock_flow

        creds = load_credentials(seed_scopes, token_path_str, "credentials.json")

    mock_path_cls.assert_called_once_with(token_path_str)
    mock_flow_cls.from_client_secrets_file.assert_called_once_with(
        "credentials.json", seed_scopes
    )
    assert creds is new_creds
