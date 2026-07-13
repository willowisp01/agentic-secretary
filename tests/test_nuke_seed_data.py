from unittest.mock import MagicMock

import pytest

from nuke_seed_data import list_all_event_ids, list_all_message_ids, nuke


def test_list_all_message_ids_single_page():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}, {"id": "m2"}]
    }
    gmail_service.users.return_value.messages.return_value.list_next.return_value = None

    ids = list_all_message_ids(gmail_service)

    assert ids == ["m1", "m2"]


def test_list_all_message_ids_follows_pagination():
    gmail_service = MagicMock()
    messages = gmail_service.users.return_value.messages.return_value

    first_request = MagicMock()
    first_request.execute.return_value = {
        "messages": [{"id": "m1"}],
        "nextPageToken": "page2",
    }
    second_request = MagicMock()
    second_request.execute.return_value = {"messages": [{"id": "m2"}]}

    messages.list.return_value = first_request
    messages.list_next.side_effect = [second_request, None]

    ids = list_all_message_ids(gmail_service)

    assert ids == ["m1", "m2"]
    assert messages.list_next.call_count == 2


def test_list_all_message_ids_handles_empty_mailbox():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {}
    gmail_service.users.return_value.messages.return_value.list_next.return_value = None

    assert list_all_message_ids(gmail_service) == []


def test_list_all_event_ids_single_page():
    calendar_service = MagicMock()
    calendar_service.events.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "e1"}, {"id": "e2"}]
    }
    calendar_service.events.return_value.list_next.return_value = None

    ids = list_all_event_ids(calendar_service)

    assert ids == ["e1", "e2"]


def test_list_all_event_ids_follows_pagination():
    calendar_service = MagicMock()
    events = calendar_service.events.return_value

    first_request = MagicMock()
    first_request.execute.return_value = {
        "items": [{"id": "e1"}],
        "nextPageToken": "page2",
    }
    second_request = MagicMock()
    second_request.execute.return_value = {"items": [{"id": "e2"}]}

    events.list.return_value = first_request
    events.list_next.side_effect = [second_request, None]

    ids = list_all_event_ids(calendar_service)

    assert ids == ["e1", "e2"]
    assert events.list_next.call_count == 2


def test_nuke_trashes_every_message_and_deletes_every_event():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}, {"id": "m2"}]
    }
    gmail_service.users.return_value.messages.return_value.list_next.return_value = None

    calendar_service = MagicMock()
    calendar_service.events.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "e1"}]
    }
    calendar_service.events.return_value.list_next.return_value = None

    nuke(gmail_service, calendar_service)

    messages = gmail_service.users.return_value.messages.return_value
    assert messages.trash.call_count == 2
    messages.trash.assert_any_call(userId="me", id="m1")
    messages.trash.assert_any_call(userId="me", id="m2")

    events = calendar_service.events.return_value
    events.delete.assert_called_once_with(calendarId="primary", eventId="e1")


def test_nuke_never_permanently_deletes_messages():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": [{"id": "m1"}]
    }
    gmail_service.users.return_value.messages.return_value.list_next.return_value = None

    calendar_service = MagicMock()
    calendar_service.events.return_value.list.return_value.execute.return_value = {}
    calendar_service.events.return_value.list_next.return_value = None

    nuke(gmail_service, calendar_service)

    messages = gmail_service.users.return_value.messages.return_value
    messages.delete.assert_not_called()


def test_nuke_never_inserts_or_sends():
    gmail_service = MagicMock()
    gmail_service.users.return_value.messages.return_value.list.return_value.execute.return_value = {}
    gmail_service.users.return_value.messages.return_value.list_next.return_value = None

    calendar_service = MagicMock()
    calendar_service.events.return_value.list.return_value.execute.return_value = {}
    calendar_service.events.return_value.list_next.return_value = None

    nuke(gmail_service, calendar_service)

    messages = gmail_service.users.return_value.messages.return_value
    messages.send.assert_not_called()
    messages.insert.assert_not_called()
    calendar_service.events.return_value.insert.assert_not_called()


def test_confirm_nuke_counts_proceeds_when_user_confirms(monkeypatch):
    from nuke_seed_data import _confirm_counts

    monkeypatch.setattr("builtins.input", lambda _: "y")

    _confirm_counts(message_count=3, event_count=2)


def test_confirm_nuke_counts_aborts_when_user_does_not_confirm(monkeypatch):
    from nuke_seed_data import _confirm_counts

    monkeypatch.setattr("builtins.input", lambda _: "n")

    with pytest.raises(SystemExit):
        _confirm_counts(message_count=3, event_count=2)
