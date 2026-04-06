from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import asyncio


def test_judge_builds_correct_prompt():
    from gateway.judge import _build_judge_prompt

    outputs = {
        "whisper_large_v3": "hello world",
        "parakeet_tdt": "hello world",
        "canary_qwen": "hello world",
        "cohere_transcribe": "hello world",
    }

    prompt = _build_judge_prompt(outputs)
    assert "Whisper Large v3" in prompt
    assert "Parakeet TDT" in prompt
    assert "Canary Qwen" in prompt
    assert "Cohere Transcribe" in prompt
    assert "hello world" in prompt
    assert "primary_model" in prompt


def test_judge_skips_none_outputs_in_prompt():
    from gateway.judge import _build_judge_prompt

    outputs = {
        "whisper_large_v3": None,
        "parakeet_tdt": "hello world",
        "canary_qwen": "hello world",
        "cohere_transcribe": None,
    }

    prompt = _build_judge_prompt(outputs)
    assert "Whisper Large v3" not in prompt
    assert "Parakeet TDT" in prompt


def test_judge_returns_transcription_confidence_and_primary_model():
    from gateway.judge import judge_transcriptions

    outputs = {
        "whisper_large_v3": "hello world",
        "parakeet_tdt": "hello world",
        "canary_qwen": "hello world",
        "cohere_transcribe": "hello world",
    }

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"transcription": "hello world", "confidence": "high", "primary_model": "parakeet_tdt"}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("gateway.judge.anthropic.Anthropic", return_value=mock_client):
        result = judge_transcriptions(outputs)

    assert result["transcription"] == "hello world"
    assert result["confidence"] == "high"
    assert result["primary_model"] == "parakeet_tdt"


def test_judge_fallback_on_api_failure():
    from gateway.judge import judge_transcriptions

    outputs = {
        "whisper_large_v3": "hello world",
        "parakeet_tdt": "best transcription here",
        "canary_qwen": "hello world",
        "cohere_transcribe": "hello world",
    }

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")

    with patch("gateway.judge.anthropic.Anthropic", return_value=mock_client):
        result = judge_transcriptions(outputs)

    # Falls back to parakeet_tdt (highest ranked model)
    assert result["transcription"] == "best transcription here"
    assert result["confidence"] == "low"
    assert result["primary_model"] == "parakeet_tdt"
