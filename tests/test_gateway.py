import io
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from gateway.main import app
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    return "test-key-123"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_transcribe_missing_api_key(client):
    response = client.post("/transcribe")
    assert response.status_code == 401


def test_transcribe_invalid_api_key(client):
    with patch("gateway.main.validate_api_key", return_value=None):
        response = client.post(
            "/transcribe",
            headers={"X-API-Key": "bad-key"},
            files={"audio": ("test.wav", b"fake-audio", "audio/wav")},
        )
    assert response.status_code == 401


def test_transcribe_missing_audio_file(client):
    with patch("gateway.main.validate_api_key", return_value="ios_app"):
        response = client.post(
            "/transcribe",
            headers={"X-API-Key": "good-key"},
        )
    assert response.status_code == 422


def test_transcribe_success(client):
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
                response = client.post(
                    "/transcribe",
                    headers={"X-API-Key": "good-key"},
                    files={"audio": ("test.wav", b"fake-audio", "audio/wav")},
                )

    assert response.status_code == 200
    data = response.json()
    assert data["transcription"] == "hello world"
    assert data["confidence"] == "high"
    assert "model_outputs" in data


def test_transcribe_runpod_failure_returns_502(client):
    with patch("gateway.main.validate_api_key", return_value="ios_app"):
        with patch("gateway.main.send_audio_to_worker", side_effect=RuntimeError("RunPod down")):
            response = client.post(
                "/transcribe",
                headers={"X-API-Key": "good-key"},
                files={"audio": ("test.wav", b"fake-audio", "audio/wav")},
            )

    assert response.status_code == 502
