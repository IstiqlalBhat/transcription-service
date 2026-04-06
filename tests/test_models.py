from unittest.mock import patch, MagicMock
import pytest


def test_transcribe_all_returns_four_keys():
    from worker.models import transcribe_all

    fake_wav = b"RIFF" + b"\x00" * 100  # dummy bytes

    with patch("worker.models._transcribe_whisper", return_value="hello world"):
        with patch("worker.models._transcribe_parakeet", return_value="hello world"):
            with patch("worker.models._transcribe_canary", return_value="hello world"):
                with patch("worker.models._transcribe_cohere", return_value="hello world"):
                    result = transcribe_all(fake_wav)

    assert set(result.keys()) == {
        "whisper_large_v3",
        "parakeet_tdt",
        "canary_qwen",
        "cohere_transcribe",
    }
    assert all(isinstance(v, str) for v in result.values())


def test_transcribe_all_handles_single_model_failure():
    from worker.models import transcribe_all

    fake_wav = b"RIFF" + b"\x00" * 100

    with patch("worker.models._transcribe_whisper", side_effect=Exception("GPU OOM")):
        with patch("worker.models._transcribe_parakeet", return_value="hello"):
            with patch("worker.models._transcribe_canary", return_value="hello"):
                with patch("worker.models._transcribe_cohere", return_value="hello"):
                    result = transcribe_all(fake_wav)

    assert result["whisper_large_v3"] is None
    assert result["parakeet_tdt"] == "hello"
    assert result["canary_qwen"] == "hello"
    assert result["cohere_transcribe"] == "hello"
