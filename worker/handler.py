import base64
import logging

from audio_utils import normalize_audio
from models import load_models, transcribe_all, _transcribe_parakeet, _parakeet_model

logger = logging.getLogger(__name__)


def handler(event: dict) -> dict:
    """RunPod serverless handler.

    Supports two modes via input.mode:
    - "fast" (default): Run Parakeet only for instant transcription (~2-3s)
    - "full": Run all 4 models for ensemble judging (~15-30s)
    """
    input_data = event.get("input", {})
    audio_b64 = input_data.get("audio")
    mode = input_data.get("mode", "fast")

    if not audio_b64:
        return {"error": "Missing 'audio' field in input. Send base64-encoded audio."}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {"error": f"Invalid base64 audio: {e}"}

    try:
        wav_bytes = normalize_audio(audio_bytes)
    except Exception as e:
        return {"error": f"Audio normalization failed: {e}"}

    if mode == "fast":
        # Fast path: Parakeet only
        if _parakeet_model is None:
            return {"error": "Parakeet model not loaded"}
        try:
            text = _transcribe_parakeet(wav_bytes)
            return {
                "mode": "fast",
                "transcription": text,
                "model_outputs": {"parakeet_tdt": text},
                "models_succeeded": 1,
            }
        except Exception as e:
            logger.error(f"Parakeet failed: {e}")
            return {"error": f"Transcription failed: {e}"}
    else:
        # Full path: all 4 models
        model_outputs = transcribe_all(wav_bytes)
        succeeded = sum(1 for v in model_outputs.values() if v is not None)
        return {
            "mode": "full",
            "model_outputs": model_outputs,
            "models_succeeded": succeeded,
        }


# RunPod entrypoint — loads models on cold start, then serves requests
if __name__ == "__main__":
    import runpod

    load_models()
    runpod.serverless.start({"handler": handler})
