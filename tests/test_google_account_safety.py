from unittest.mock import MagicMock

import pytest

from _google_account_safety import confirm_target_account


def test_confirm_target_account_returns_email_when_user_confirms(monkeypatch):
    gmail_service = MagicMock()
    gmail_service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "burner@example.com"
    }
    monkeypatch.setattr("builtins.input", lambda _: "y")

    result = confirm_target_account(gmail_service, action="seed demo data")

    assert result == "burner@example.com"


def test_confirm_target_account_aborts_when_user_does_not_confirm(monkeypatch):
    gmail_service = MagicMock()
    gmail_service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "burner@example.com"
    }
    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(SystemExit):
        confirm_target_account(gmail_service, action="seed demo data")


def test_confirm_target_account_prompt_reflects_given_action(monkeypatch):
    gmail_service = MagicMock()
    gmail_service.users.return_value.getProfile.return_value.execute.return_value = {
        "emailAddress": "burner@example.com"
    }
    seen_prompts = []

    def fake_input(prompt):
        seen_prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)

    confirm_target_account(gmail_service, action="trash and delete all demo data")

    assert len(seen_prompts) == 1
    assert "trash and delete all demo data" in seen_prompts[0]
    assert "seed demo data" not in seen_prompts[0]
