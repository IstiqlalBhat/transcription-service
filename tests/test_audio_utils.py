import subprocess
import struct
import wave
import io
import pytest


def _make_wav_bytes(sample_rate: int, num_channels: int, duration_s: float) -> bytes:
    """Generate a silent WAV file as bytes."""
    num_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames * num_channels)
    return buf.getvalue()


def test_converts_stereo_to_mono():
    from worker.audio_utils import normalize_audio

    stereo_wav = _make_wav_bytes(sample_rate=44100, num_channels=2, duration_s=0.5)
    result = normalize_audio(stereo_wav)

    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_resamples_to_16khz():
    from worker.audio_utils import normalize_audio

    high_rate_wav = _make_wav_bytes(sample_rate=48000, num_channels=1, duration_s=0.5)
    result = normalize_audio(high_rate_wav)

    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getframerate() == 16000


def test_already_normalized_passes_through():
    from worker.audio_utils import normalize_audio

    correct_wav = _make_wav_bytes(sample_rate=16000, num_channels=1, duration_s=0.5)
    result = normalize_audio(correct_wav)

    buf = io.BytesIO(result)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_empty_audio_raises():
    from worker.audio_utils import normalize_audio

    with pytest.raises(ValueError, match="empty"):
        normalize_audio(b"")
