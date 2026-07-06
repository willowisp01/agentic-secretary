from unittest.mock import MagicMock, patch

from agentic_secretary.auth import get_credentials


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
