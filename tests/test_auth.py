from unittest.mock import patch
import pytest


def test_valid_key_returns_app_name():
    from gateway.auth import validate_api_key

    keys = {"ios_app": "secret-key-123", "web_app": "secret-key-456"}

    with patch("gateway.auth.settings") as mock_settings:
        mock_settings.get_api_keys.return_value = keys
        result = validate_api_key("secret-key-123")

    assert result == "ios_app"


def test_invalid_key_returns_none():
    from gateway.auth import validate_api_key

    keys = {"ios_app": "secret-key-123"}

    with patch("gateway.auth.settings") as mock_settings:
        mock_settings.get_api_keys.return_value = keys
        result = validate_api_key("wrong-key")

    assert result is None


def test_empty_key_returns_none():
    from gateway.auth import validate_api_key

    with patch("gateway.auth.settings") as mock_settings:
        mock_settings.get_api_keys.return_value = {}
        result = validate_api_key("")

    assert result is None
