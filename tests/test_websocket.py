import base64
import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from gateway.main import app
    return TestClient(app)


def test_websocket_rejects_missing_api_key(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/transcribe"):
            pass


def test_websocket_rejects_invalid_api_key(client):
    with patch("gateway.main.validate_api_key", return_value=None):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/transcribe?api_key=bad-key"):
                pass


def test_websocket_accepts_valid_key_and_transcribes(client):
    mock_worker_response = {
        "model_outputs": {
            "whisper_large_v3": "hello world",
            "parakeet_tdt": "hello world",
            "canary_qwen": "hello world",
            "cohere_transcribe": "hello world",
        },
        "models_succeeded": 4,
    }

    mock_judge_response = {
        "transcription": "hello world",
        "confidence": "high",
        "primary_model": "parakeet_tdt",
    }

    with patch("gateway.main.validate_api_key", return_value="ios_app"):
        with patch("gateway.main.send_audio_to_worker", return_value=mock_worker_response):
            with patch("gateway.main.judge_transcriptions", return_value=mock_judge_response):
                with client.websocket_connect("/ws/transcribe?api_key=good-key") as ws:
                    ws.send_bytes(b"fake-audio-chunk")
                    response = ws.receive_json()

    assert response["transcription"] == "hello world"
    assert response["confidence"] == "high"
    assert "model_outputs" in response
