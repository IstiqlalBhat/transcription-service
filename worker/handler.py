import base64
import logging

from worker.audio_utils import normalize_audio
from worker.models import load_models, transcribe_all

logger = logging.getLogger(__name__)


def handler(event: dict) -> dict:
    """RunPod serverless handler. Receives base64 audio, returns 4 transcriptions."""
    input_data = event.get("input", {})
    audio_b64 = input_data.get("audio")

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

    model_outputs = transcribe_all(wav_bytes)

    succeeded = sum(1 for v in model_outputs.values() if v is not None)

    return {
        "model_outputs": model_outputs,
        "models_succeeded": succeeded,
    }


# RunPod entrypoint — loads models on cold start, then serves requests
if __name__ == "__main__":
    import runpod  # noqa: F401 — only needed at runtime, not imported at module level

    load_models()
    runpod.serverless.start({"handler": handler})
