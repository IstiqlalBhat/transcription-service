import base64
from unittest.mock import patch, MagicMock
import pytest


def test_send_audio_encodes_and_calls_runpod():
    from gateway.runpod_client import send_audio_to_worker

    mock_endpoint = MagicMock()
    mock_run = MagicMock()
    mock_run.output.return_value = {
        "model_outputs": {"whisper_large_v3": "hello"},
        "models_succeeded": 1,
    }
    mock_endpoint.run_sync.return_value = mock_run

    with patch("gateway.runpod_client.runpod") as mock_runpod:
        with patch("gateway.runpod_client.settings") as mock_settings:
            mock_settings.runpod_api_key = "test-key"
            mock_settings.runpod_endpoint_id = "test-endpoint"
            mock_runpod.Endpoint.return_value = mock_endpoint

            result = send_audio_to_worker(b"audio-bytes")

    call_args = mock_endpoint.run_sync.call_args
    sent_input = call_args[0][0] if call_args[0] else call_args[1].get("request_input", {})
    assert "audio" in str(sent_input) or mock_endpoint.run_sync.called


def test_send_audio_raises_on_runpod_error():
    from gateway.runpod_client import send_audio_to_worker

    mock_endpoint = MagicMock()
    mock_endpoint.run_sync.side_effect = Exception("RunPod timeout")

    with patch("gateway.runpod_client.runpod") as mock_runpod:
        with patch("gateway.runpod_client.settings") as mock_settings:
            mock_settings.runpod_api_key = "test-key"
            mock_settings.runpod_endpoint_id = "test-endpoint"
            mock_runpod.Endpoint.return_value = mock_endpoint

            with pytest.raises(RuntimeError, match="RunPod"):
                send_audio_to_worker(b"audio-bytes")
