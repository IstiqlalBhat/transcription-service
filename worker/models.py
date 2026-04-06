import io
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

# Global model references — loaded once on cold start
_whisper_model = None
_whisper_processor = None
_parakeet_model = None
_canary_model = None
_cohere_model = None
_cohere_processor = None


def load_models():
    """Load all 4 ASR models into VRAM. Called once on worker cold start."""
    global _whisper_model, _whisper_processor
    global _parakeet_model, _canary_model
    global _cohere_model, _cohere_processor

    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import nemo.collections.asr as nemo_asr

    logger.info("Loading Whisper Large v3...")
    _whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    _whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3"
    ).to("cuda")

    logger.info("Loading Parakeet TDT 0.6b v2...")
    _parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2"
    )
    _parakeet_model = _parakeet_model.to("cuda")

    logger.info("Loading Canary Qwen 2.5b...")
    _canary_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/canary-qwen-2.5b"
    )
    _canary_model = _canary_model.to("cuda")

    logger.info("Loading Cohere Transcribe...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    _cohere_processor = AutoProcessor.from_pretrained("cohere-transcribe-03-2026")
    _cohere_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "cohere-transcribe-03-2026"
    ).to("cuda")

    logger.info("All models loaded.")


def _wav_bytes_to_array(wav_bytes: bytes):
    """Convert WAV bytes (16kHz mono) to float32 numpy array."""
    import numpy as np  # noqa: F401 — imported for type usage by callers
    import soundfile as sf

    audio_array, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return audio_array


def _transcribe_whisper(wav_bytes: bytes) -> str:
    import torch

    audio = _wav_bytes_to_array(wav_bytes)
    inputs = _whisper_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        predicted_ids = _whisper_model.generate(**inputs)
    return _whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()


def _transcribe_parakeet(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        f.flush()
        try:
            results = _parakeet_model.transcribe([f.name])
            return results[0].strip() if isinstance(results[0], str) else results[0].text.strip()
        finally:
            os.unlink(f.name)


def _transcribe_canary(wav_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        f.flush()
        try:
            results = _canary_model.transcribe([f.name])
            return results[0].strip() if isinstance(results[0], str) else results[0].text.strip()
        finally:
            os.unlink(f.name)


def _transcribe_cohere(wav_bytes: bytes) -> str:
    import torch

    audio = _wav_bytes_to_array(wav_bytes)
    inputs = _cohere_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        predicted_ids = _cohere_model.generate(**inputs)
    return _cohere_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()


def transcribe_all(wav_bytes: bytes) -> dict[str, str | None]:
    """Run all 4 models and return their outputs. If a model fails, its value is None."""
    models = {
        "whisper_large_v3": _transcribe_whisper,
        "parakeet_tdt": _transcribe_parakeet,
        "canary_qwen": _transcribe_canary,
        "cohere_transcribe": _transcribe_cohere,
    }

    results = {}
    for name, fn in models.items():
        try:
            results[name] = fn(wav_bytes)
        except Exception as e:
            logger.error(f"Model {name} failed: {e}")
            results[name] = None

    return results
