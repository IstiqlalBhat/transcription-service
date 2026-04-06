import base64
from unittest.mock import patch, MagicMock
import pytest


def test_handler_returns_transcriptions():
    from worker.handler import handler

    audio_b64 = base64.b64encode(b"fake-audio-data").decode()

    mock_results = {
        "whisper_large_v3": "hello world",
        "parakeet_tdt": "hello world",
        "canary_qwen": "hello world",
        "cohere_transcribe": "hello world",
    }

    with patch("worker.handler.normalize_audio", return_value=b"normalized-wav"):
        with patch("worker.handler.transcribe_all", return_value=mock_results):
            result = handler({"input": {"audio": audio_b64}})

    assert result["model_outputs"] == mock_results
    assert result["models_succeeded"] == 4


def test_handler_missing_audio_returns_error():
    from worker.handler import handler

    result = handler({"input": {}})
    assert "error" in result


def test_handler_reports_partial_success():
    from worker.handler import handler

    audio_b64 = base64.b64encode(b"fake-audio-data").decode()

    mock_results = {
        "whisper_large_v3": None,
        "parakeet_tdt": "hello",
        "canary_qwen": "hello",
        "cohere_transcribe": None,
    }

    with patch("worker.handler.normalize_audio", return_value=b"normalized-wav"):
        with patch("worker.handler.transcribe_all", return_value=mock_results):
            result = handler({"input": {"audio": audio_b64}})

    assert result["models_succeeded"] == 2
    assert result["model_outputs"]["whisper_large_v3"] is None
